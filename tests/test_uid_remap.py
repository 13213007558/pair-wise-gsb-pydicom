"""End-to-end checks for the batch UID remapper.

Run:  python3 -m unittest tests.test_uid_remap -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.make_synthetic import (  # noqa: E402
    CODING_SCHEME, FOR_UID, IMPLEMENTATION_CLASS, PRIVATE_UI_VALUE,
    RTSTRUCT_SOP_CLASS, SC_SOP_CLASS, SERIES_IMG, SERIES_STRUCT, SOP_A,
    SOP_B, SOP_C, STUDY, build_all,
)
from uid_remap._pydicom import pydicom  # noqa: E402
from uid_remap.assigners import hash_assigner  # noqa: E402
from uid_remap.remapper import (  # noqa: E402
    RemapValidationError, _iter_ui, _tag_pair, _ui_values, is_valid_uid,
    remap_batch,
)
from uid_remap.uidtags import INSTANCE_UID_TAGS  # noqa: E402


def collect_ui(ds) -> list[tuple[tuple, tuple[int, int], list[str]]]:
    """All UI elements of one file as (path, tag, values) tuples."""
    found = []
    for scope, container in (("FileMeta", ds.file_meta), ("Dataset", ds)):
        if container is None:
            continue
        for path, name, element in _iter_ui(container):
            found.append(((scope,) + path + (name,), _tag_pair(element.tag), _ui_values(element)))
    return found


class RemapEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="uid_remap_test_"))
        cls.in_dir = cls.tmp / "in"
        cls.out_dir = cls.tmp / "out"
        cls.inputs = build_all(cls.in_dir)
        cls.result = remap_batch(cls.inputs, cls.out_dir, hash_assigner(salt="batch1"))
        cls.outputs = {p.name: pydicom.dcmread(str(p))
                       for p in sorted(cls.out_dir.glob("*.dcm"))}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_mapping_is_valid_and_covers_every_instance_uid(self):
        old_uids = set()
        for path in self.inputs:
            for _, tag, values in collect_ui(pydicom.dcmread(str(path))):
                if tag in INSTANCE_UID_TAGS:
                    old_uids.update(values)
        self.assertEqual(set(self.result.mapping), old_uids)
        for old, new in self.result.mapping.items():
            self.assertNotEqual(old, new)
            self.assertTrue(is_valid_uid(new), new)
        self.assertEqual(len(set(self.result.mapping.values())), len(self.result.mapping))

    def test_no_old_instance_uid_survives(self):
        old = set(self.result.mapping)
        for ds in self.outputs.values():
            for _, tag, values in collect_ui(ds):
                if tag in INSTANCE_UID_TAGS:
                    self.assertFalse(old & set(values))

    def test_references_stay_closed(self):
        mapping = self.result.mapping
        sop_uids, referenced = set(), set()
        series, series_refs = set(), set()
        fors, for_refs = set(), set()
        studies = set()
        for ds in self.outputs.values():
            for path, tag, values in collect_ui(ds):
                joined = "/".join(path)
                if tag == (0x0008, 0x0018):
                    sop_uids.update(values)
                elif tag == (0x0008, 0x1155):
                    referenced.update(values)
                elif tag == (0x0020, 0x000E):
                    (series_refs if "RTReferencedSeriesSequence" in joined else series).update(values)
                elif tag == (0x0020, 0x0052):
                    (for_refs if "ReferencedFrameOfReferenceSequence" in joined else fors).update(values)
                elif tag == (0x0020, 0x000D):
                    studies.update(values)
        self.assertEqual(sop_uids, {mapping[SOP_A], mapping[SOP_B], mapping[SOP_C]})
        self.assertTrue(referenced <= sop_uids, referenced - sop_uids)
        self.assertEqual(series_refs, {mapping[SERIES_IMG]})
        self.assertEqual(series, {mapping[SERIES_IMG], mapping[SERIES_STRUCT]})
        self.assertEqual(for_refs, {mapping[FOR_UID]})
        self.assertTrue(for_refs <= fors)
        self.assertEqual(studies, {mapping[STUDY]})

    def test_file_meta_synced_with_dataset(self):
        expected = {"img_a.dcm": SOP_A, "img_b.dcm": SOP_B, "struct.dcm": SOP_C}
        for name, ds in self.outputs.items():
            self.assertEqual(str(ds.file_meta.MediaStorageSOPInstanceUID),
                             self.result.mapping[expected[name]])
            self.assertEqual(str(ds.file_meta.MediaStorageSOPInstanceUID),
                             str(ds.SOPInstanceUID))

    def test_definition_uids_untouched(self):
        keep = {SC_SOP_CLASS, RTSTRUCT_SOP_CLASS, CODING_SCHEME, IMPLEMENTATION_CLASS}
        texts = set()
        for ds in self.outputs.values():
            for _, _, values in collect_ui(ds):
                texts.update(values)
        self.assertTrue(keep <= texts, keep - texts)
        self.assertFalse(set(self.result.mapping) & keep)
        for ds in self.outputs.values():
            self.assertEqual(str(ds.file_meta.TransferSyntaxUID),
                             "1.2.840.10008.1.2.1")

    def test_private_ui_reported_not_modified(self):
        self.assertEqual(len(self.result.unclassified), 1)
        entry = self.result.unclassified[0]
        self.assertEqual(entry["file"], "struct.dcm")
        self.assertEqual(entry["tag"], "(0009,1001)")
        self.assertEqual(entry["value"], PRIVATE_UI_VALUE)
        struct = self.outputs["struct.dcm"]
        texts = set()
        for _, _, values in collect_ui(struct):
            texts.update(values)
        self.assertIn(PRIVATE_UI_VALUE, texts)

    def test_change_list_matches_mapping(self):
        self.assertTrue(self.result.changes)
        for change in self.result.changes:
            for old, new in zip(change.old.split("\\"), change.new.split("\\")):
                self.assertEqual(self.result.mapping[old], new)

    def test_mapping_reuse_keeps_identities_stable(self):
        out2 = self.tmp / "out2"
        result2 = remap_batch(self.inputs, out2, hash_assigner(salt="OTHER"),
                              mapping_in=self.result.mapping)
        self.assertEqual(result2.mapping, self.result.mapping)
        for name in ("img_a.dcm", "img_b.dcm", "struct.dcm"):
            self.assertEqual((self.out_dir / name).read_bytes(),
                             (out2 / name).read_bytes())

    def test_conflicting_new_uids_abort_before_writing(self):
        out = self.tmp / "conflict_out"
        with self.assertRaises(RemapValidationError) as ctx:
            remap_batch(self.inputs, out, lambda old: "2.25.1")
        self.assertIn("conflict", str(ctx.exception))
        self.assertFalse(out.exists())

    def test_invalid_new_uid_aborts(self):
        with self.assertRaises(RemapValidationError):
            remap_batch(self.inputs, self.tmp / "x1", lambda old: "9" * 65)
        with self.assertRaises(RemapValidationError):
            remap_batch(self.inputs, self.tmp / "x2", lambda old: "1.2.03.4")

    def test_unstable_assign_aborts(self):
        counter = {"n": 0}
        def flaky(old):
            counter["n"] += 1
            return "2.25." + str(counter["n"])
        with self.assertRaises(RemapValidationError) as ctx:
            remap_batch(self.inputs, self.tmp / "x3", flaky)
        self.assertIn("unstable", str(ctx.exception))

    def test_roundtrip_other_transfer_syntaxes(self):
        from pydicom.filewriter import dcmwrite
        from pydicom.uid import ImplicitVRLittleEndian, ExplicitVRBigEndian
        for ts in (ImplicitVRLittleEndian, ExplicitVRBigEndian):
            src = pydicom.dcmread(str(self.in_dir / "img_a.dcm"))
            variant = self.tmp / ("variant_" + str(ts).rsplit(".", 1)[-1] + ".dcm")
            src.file_meta.TransferSyntaxUID = ts
            dcmwrite(str(variant), src,
                     implicit_vr=(ts == ImplicitVRLittleEndian),
                     little_endian=(ts != ExplicitVRBigEndian),
                     enforce_file_format=True)
            out = self.tmp / ("out_" + str(ts).rsplit(".", 1)[-1])
            result = remap_batch([variant], out, hash_assigner(salt="ts"))
            reread = pydicom.dcmread(str(out / variant.name))
            self.assertEqual(str(reread.file_meta.TransferSyntaxUID), str(ts))
            self.assertEqual(str(reread.StudyInstanceUID), result.mapping[STUDY])


if __name__ == "__main__":
    unittest.main(verbosity=2)
