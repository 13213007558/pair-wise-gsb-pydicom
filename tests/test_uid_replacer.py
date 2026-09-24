"""Tests for pydicom.uid_replacer."""

import json
import random

import pytest

from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
from pydicom.uid_replacer import (
    DEFINITIONAL,
    INSTANCE,
    UNCLASSIFIED,
    UIDRemapError,
    UIDRemapper,
    classify_ui_element,
    load_mapping,
    validate_uid,
)

OLD_STUDY = "1.2.3.100"
OLD_SERIES = "1.2.3.100.1"
OLD_FOR = "1.2.3.100.2"
OLD_SOP = ["1.2.3.100.1.1", "1.2.3.100.1.2", "1.2.3.100.1.3"]
OLD_PRIVATE = "1.2.3.4.5.6.7.8.9"
OLD_SELECTOR = "1.2.3.4.999"


def make_allocator(start=1):
    """Pure allocator: same old UID always gets the same new UID."""
    assigned = {}

    def allocate(old):
        if old not in assigned:
            assigned[old] = f"1.2.826.0.1.3680043.9.7437.{len(assigned) + start}"
        return assigned[old]

    return allocate


def make_file(path, sop_uid, refs):
    meta = FileMetaDataset()
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.3.4.5.6.7"

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = OLD_STUDY
    ds.SeriesInstanceUID = OLD_SERIES
    ds.FrameOfReferenceUID = OLD_FOR
    ds.Modality = "CT"
    ds.PatientID = "SYNTH"

    items = []
    for ref_sop in refs:
        item = Dataset()
        item.ReferencedSOPClassUID = CTImageStorage
        item.ReferencedSOPInstanceUID = ref_sop
        items.append(item)
    ds.ReferencedImageSequence = items

    # Private UI element: must be reported, not rewritten.
    ds.add_new(0x00090010, "LO", "ACME")
    ds.add_new(0x00091001, "UI", OLD_PRIVATE)
    # Public UI element with unknown role: also reported, not rewritten.
    ds.SelectorUIValue = OLD_SELECTOR

    ds.save_as(
        path,
        enforce_file_format=True,
        little_endian=True,
        implicit_vr=False,
    )
    return path


@pytest.fixture
def three_files(tmp_path):
    """Three mutually referencing synthetic files."""
    src = tmp_path / "src"
    src.mkdir()
    paths = [
        make_file(src / "img1.dcm", OLD_SOP[0], [OLD_SOP[1], OLD_SOP[2]]),
        make_file(src / "img2.dcm", OLD_SOP[1], [OLD_SOP[2], OLD_SOP[0]]),
        make_file(src / "img3.dcm", OLD_SOP[2], [OLD_SOP[0]]),
    ]
    return paths


def instance_ui_values(ds):
    """All values of instance-classified UI elements in one dataset."""
    from pydicom.uid_replacer import _iter_ui_values

    found = []
    sources = [ds]
    if ds.file_meta is not None:
        sources.append(ds.file_meta)
    for source in sources:
        for element, tag, index, keyword, _ in _iter_ui_values(source):
            if classify_ui_element(tag, keyword) != INSTANCE:
                continue
            values = element.value
            if index is not None:
                found.append((keyword, str(values[index])))
            elif isinstance(values, (list, tuple)):
                found.extend((keyword, str(v)) for v in values)
            else:
                found.append((keyword, str(values)))
    return found


class TestClassification:
    def test_definitional_uids_are_kept(self):
        assert classify_ui_element(0x00080016, "SOPClassUID") == DEFINITIONAL
        assert classify_ui_element(0x00020002, "TransferSyntaxUID") == DEFINITIONAL
        assert classify_ui_element(0x0008010C, "CodingSchemeUID") == DEFINITIONAL
        assert classify_ui_element(0x00081150, "ReferencedSOPClassUID") == DEFINITIONAL

    def test_instance_uids_are_replaced(self):
        assert classify_ui_element(0x0020000D, "StudyInstanceUID") == INSTANCE
        assert classify_ui_element(0x0020000E, "SeriesInstanceUID") == INSTANCE
        assert classify_ui_element(0x00080018, "SOPInstanceUID") == INSTANCE
        assert classify_ui_element(0x00020003, "MediaStorageSOPInstanceUID") == INSTANCE
        assert classify_ui_element(0x00081155, "ReferencedSOPInstanceUID") == INSTANCE
        assert classify_ui_element(0x00200052, "FrameOfReferenceUID") == INSTANCE

    def test_private_and_unknown_are_unclassified(self):
        assert classify_ui_element(0x00091001, "<unknown>") == UNCLASSIFIED
        assert classify_ui_element(0x0072007F, "SelectorUIValue") == UNCLASSIFIED


class TestValidateUID:
    def test_valid(self):
        assert validate_uid("1.2.826.0.1.3680043.9.7437.1") is None
        assert validate_uid("0.0.0") is None

    def test_too_long(self):
        assert validate_uid("1." + "2" * 64) is not None

    def test_bad_characters(self):
        assert validate_uid("1.2.abc") is not None
        assert validate_uid("1.2..3") is not None
        assert validate_uid("1.02.3") is not None


class TestRemap:
    def test_reference_closure(self, three_files, tmp_path):
        out = tmp_path / "out"
        remapper = UIDRemapper(allocator=make_allocator())
        report = remapper.remap(three_files, out)

        datasets = [dcmread(out / p.name) for p in three_files]

        # Identity UIDs changed, definitional UIDs untouched.
        old_identities = set(OLD_SOP) | {OLD_STUDY, OLD_SERIES, OLD_FOR}
        new_sop_uids = set()
        for ds in datasets:
            assert ds.SOPClassUID == CTImageStorage
            assert ds.file_meta.MediaStorageSOPClassUID == CTImageStorage
            assert ds.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
            assert ds.file_meta.ImplementationClassUID == "1.2.3.4.5.6.7"

            # File Meta stays in sync with the dataset body.
            assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
            new_sop_uids.add(ds.SOPInstanceUID)

            for keyword, value in instance_ui_values(ds):
                assert value not in old_identities, (keyword, value)

        # Same old identity -> same new identity across files.
        study_uids = {ds.StudyInstanceUID for ds in datasets}
        series_uids = {ds.SeriesInstanceUID for ds in datasets}
        for_uids = {ds.FrameOfReferenceUID for ds in datasets}
        assert len(study_uids) == len(series_uids) == len(for_uids) == 1

        # References are closed: every referenced SOP Instance UID is one
        # of the new SOP Instance UIDs of the batch.
        referenced = set()
        for ds in datasets:
            for item in ds.ReferencedImageSequence:
                assert item.ReferencedSOPClassUID == CTImageStorage
                referenced.add(item.ReferencedSOPInstanceUID)
        assert referenced == new_sop_uids

        # Mapping is injective and covers every old identity.
        mapping = report.mapping
        assert old_identities <= set(mapping)
        assert len(set(mapping.values())) == len(mapping)

        # Change list records every rewrite.
        changed_old = {c.old for c in report.changes}
        assert old_identities <= changed_old
        for change in report.changes:
            assert mapping[change.old] == change.new

        # Private and unknown UI fields are reported and left unchanged.
        reported = {(u.tag, u.value) for u in report.unclassified}
        assert ("(0009,1001)", OLD_PRIVATE) in reported
        assert ("(0072,007F)", OLD_SELECTOR) in reported
        for ds in datasets:
            assert ds[0x00091001].value == OLD_PRIVATE
            assert ds.SelectorUIValue == OLD_SELECTOR

    def test_mapping_export_and_reuse(self, three_files, tmp_path):
        out1 = tmp_path / "out1"
        report = UIDRemapper(allocator=make_allocator()).remap(three_files, out1)
        mapping_file = tmp_path / "mapping.json"
        report.export_mapping(mapping_file)

        loaded = load_mapping(mapping_file)
        assert loaded == report.mapping
        assert json.loads(mapping_file.read_text())["old_to_new"] == loaded

        # Next batch: a new file in the same study keeps the study UID
        # from the exported mapping, new identities go to the allocator.
        new_sop = "1.2.3.100.1.4"
        extra = make_file(tmp_path / "src" / "img4.dcm", new_sop, [OLD_SOP[0]])
        out2 = tmp_path / "out2"
        report2 = UIDRemapper(
            allocator=make_allocator(start=1000), seed_mapping=loaded
        ).remap([extra], out2)

        ds = dcmread(out2 / extra.name)
        assert ds.StudyInstanceUID == loaded[OLD_STUDY]
        assert ds.SeriesInstanceUID == loaded[OLD_SERIES]
        assert (
            ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID
            == (loaded[OLD_SOP[0]])
        )
        assert report2.mapping[new_sop] not in loaded.values()

    def test_conflicting_new_uids_rejected(self, three_files, tmp_path):
        out = tmp_path / "out"
        remapper = UIDRemapper(allocator=lambda old: "1.2.3.4")
        with pytest.raises(UIDRemapError) as exc:
            remapper.remap(three_files, out)
        assert any("assigned to both" in p for p in exc.value.problems)
        assert not list(out.glob("*"))

    def test_invalid_new_uid_rejected(self, three_files, tmp_path):
        out = tmp_path / "out"
        remapper = UIDRemapper(allocator=lambda old: "1.2." + "3" * 64)
        with pytest.raises(UIDRemapError) as exc:
            remapper.remap(three_files, out)
        assert any("characters long" in p for p in exc.value.problems)
        assert not list(out.glob("*"))

    def test_unstable_allocator_rejected(self, three_files, tmp_path):
        out = tmp_path / "out"
        rng = random.Random(42)

        def unstable(old):
            return f"1.2.3.{rng.randrange(10**6)}"

        remapper = UIDRemapper(allocator=unstable)
        with pytest.raises(UIDRemapError) as exc:
            remapper.remap(three_files, out)
        assert any("repeated calls" in p for p in exc.value.problems)
        assert not list(out.glob("*"))

    def test_missing_mapping_rejected(self, three_files, tmp_path):
        out = tmp_path / "out"
        remapper = UIDRemapper(seed_mapping={OLD_STUDY: "9.9.9.1"})
        with pytest.raises(UIDRemapError) as exc:
            remapper.remap(three_files, out)
        assert any("not in the seed mapping" in p for p in exc.value.problems)

    def test_meta_body_mismatch_warns(self, tmp_path):
        path = make_file(tmp_path / "odd.dcm", OLD_SOP[0], [])
        ds = dcmread(path)
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.100.1.99"
        ds.save_as(path, enforce_file_format=False)

        out = tmp_path / "out"
        report = UIDRemapper(allocator=make_allocator()).remap([path], out)
        assert any("differs from SOPInstanceUID" in w for w in report.warnings)
