"""Build three purely synthetic, mutually referencing DICOM files.

- img_a.dcm / img_b.dcm: two "images" sharing Study ST1, Series SE1 and
  FrameOfReference F1. img_a also references img_b through
  ReferencedImageSequence.
- struct.dcm: a "structure set" in its own Series SE2 (same Study ST1)
  that references F1, SE1 and both SOP instances through a three-level
  nested sequence chain, carries a CodingSchemeUID (definition class),
  a multi-valued FailedSOPInstanceUIDList, and a private UI tag that no
  rule covers (must be reported, not modified).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uid_remap._pydicom import pydicom  # noqa: E402
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

# Old identities (shared across the set).
STUDY = "1.2.826.0.1.3680043.9.7433.1.1"
SERIES_IMG = "1.2.826.0.1.3680043.9.7433.1.2"
SERIES_STRUCT = "1.2.826.0.1.3680043.9.7433.1.3"
FOR_UID = "1.2.826.0.1.3680043.9.7433.1.4"
SOP_A = "1.2.826.0.1.3680043.9.7433.1.5"
SOP_B = "1.2.826.0.1.3680043.9.7433.1.6"
SOP_C = "1.2.826.0.1.3680043.9.7433.1.7"

# Definition-class UIDs (must survive untouched).
SC_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.7"          # Secondary Capture
RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
CODING_SCHEME = "1.2.840.10008.2.16.4"              # DICOM Controlled Terminology
IMPLEMENTATION_CLASS = "2.25.12345678901234567890"
PRIVATE_UI_VALUE = "1.3.6.1.4.1.99999.1.2.3"


def _meta(sop_class: str, sop_instance: str) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_instance
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = IMPLEMENTATION_CLASS
    meta.ImplementationVersionName = "UIDREMAP1"
    return meta


def _file(sop_class: str, sop_instance: str) -> Dataset:
    ds = Dataset()
    ds.file_meta = _meta(sop_class, sop_instance)
    ds.preamble = b"\x00" * 128
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_instance
    return ds


def build_image(sop_uid: str, referenced_sop: str | None) -> Dataset:
    ds = _file(SC_SOP_CLASS, sop_uid)
    ds.SeriesDescription = "synthetic image series"
    ds.StudyInstanceUID = STUDY
    ds.SeriesInstanceUID = SERIES_IMG
    ds.FrameOfReferenceUID = FOR_UID
    if referenced_sop:
        ref = Dataset()
        ref.ReferencedSOPClassUID = SC_SOP_CLASS
        ref.ReferencedSOPInstanceUID = referenced_sop
        ds.ReferencedImageSequence = [ref]
    return ds


def build_struct() -> Dataset:
    ds = _file(RTSTRUCT_SOP_CLASS, SOP_C)
    ds.SeriesDescription = "synthetic struct series"
    ds.StudyInstanceUID = STUDY
    ds.SeriesInstanceUID = SERIES_STRUCT
    ds.FailedSOPInstanceUIDList = [SOP_A, SOP_B]      # multi-valued UI
    ds.CodingSchemeUID = CODING_SCHEME

    def contour_ref(sop_uid: str) -> Dataset:
        item = Dataset()
        item.ReferencedSOPClassUID = SC_SOP_CLASS
        item.ReferencedSOPInstanceUID = sop_uid
        return item

    series_item = Dataset()
    series_item.SeriesInstanceUID = SERIES_IMG
    series_item.ContourImageSequence = [contour_ref(SOP_A), contour_ref(SOP_B)]

    study_item = Dataset()
    study_item.StudyInstanceUID = STUDY
    study_item.RTReferencedSeriesSequence = [series_item]

    for_item = Dataset()
    for_item.FrameOfReferenceUID = FOR_UID
    for_item.RTReferencedStudySequence = [study_item]
    ds.ReferencedFrameOfReferenceSequence = [for_item]

    ds.add_new(0x00091001, "UI", PRIVATE_UI_VALUE)    # unclassified private UI
    return ds


def build_all(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "img_a.dcm": build_image(SOP_A, referenced_sop=SOP_B),
        "img_b.dcm": build_image(SOP_B, referenced_sop=None),
        "struct.dcm": build_struct(),
    }
    paths = []
    for name, ds in files.items():
        path = out_dir / name
        ds.save_as(str(path), little_endian=True, implicit_vr=False,
                   enforce_file_format=True)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "synthetic_in"
    for path in build_all(target):
        print(path)
