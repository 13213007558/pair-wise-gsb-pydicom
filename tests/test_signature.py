# Copyright 2008-2025 pydicom authors. See LICENSE file for details.
"""Tests for signature-aware edit planning (pydicom.signature)."""

from io import BytesIO

import pytest

from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
from pydicom.filewriter import dcmwrite
from pydicom.sequence import Sequence
from pydicom.signature import (
    InvalidatedSignature,
    SignatureEditPlan,
    SignatureInvalidationError,
)
from pydicom.tag import Tag

PATIENT_NAME = Tag(0x00100010)
PATIENT_ID = Tag(0x00100020)
STUDY_DATE = Tag(0x00080020)
PATIENT_BIRTH_DATE = Tag(0x00100030)
REFERENCED_SERIES_SEQUENCE = Tag(0x00081115)
STUDY_INSTANCE_UID = Tag(0x0020000D)
SERIES_INSTANCE_UID = Tag(0x0020000E)
DATA_ELEMENTS_SIGNED = Tag(0x04000020)
PRIVATE_TAG = Tag(0x00091001)


def _mac_item(mac_id, algorithm, signed_tags):
    item = Dataset()
    item.MACIDNumber = mac_id
    item.MACAlgorithm = algorithm
    item.add_new(DATA_ELEMENTS_SIGNED, "AT", list(signed_tags))
    return item


def _signature(uid, mac_id, signed_tags=None):
    item = Dataset()
    item.DigitalSignatureUID = uid
    item.MACIDNumber = mac_id
    if signed_tags is not None:
        item.add_new(DATA_ELEMENTS_SIGNED, "AT", list(signed_tags))
    item.Signature = b"\x00" * 4
    return item


def make_signed_dataset():
    """Build a dataset with overlapping top-level signatures and one
    nested signature.

    * sig A (1.2.3.1), MAC ID 1: PatientName + PatientID (own list)
    * sig B (1.2.3.2), MAC ID 2: PatientID + StudyDate +
      ReferencedSeriesSequence (coverage from the MAC parameters item)
    * sig D (1.2.3.4), MAC ID 2: StudyDate (own list, shares MAC ID 2
      with B)

    The item of ReferencedSeriesSequence is its own scope with
    sig C (1.2.3.3), MAC ID 1, covering StudyInstanceUID.
    """
    ds = Dataset()
    ds.PatientName = "Doe^John"
    ds.PatientID = "12345"
    ds.StudyDate = "20240101"
    ds.PatientBirthDate = "19800101"
    ds.add_new_private("ACME", 0x0009, 0x01, "secret", vr="LO")

    nested = Dataset()
    nested.StudyInstanceUID = "1.2.3.4.5"
    nested.SeriesInstanceUID = "1.2.3.4.6"
    nested.MACParametersSequence = Sequence(
        [_mac_item(1, "RIPEMD160", [STUDY_INSTANCE_UID])]
    )
    nested.DigitalSignaturesSequence = Sequence(
        [_signature("1.2.3.3", 1, [STUDY_INSTANCE_UID])]
    )
    ds.ReferencedSeriesSequence = Sequence([nested])

    ds.MACParametersSequence = Sequence(
        [
            _mac_item(1, "RIPEMD160", [PATIENT_NAME, PATIENT_ID]),
            _mac_item(
                2,
                "SHA1",
                [PATIENT_ID, STUDY_DATE, REFERENCED_SERIES_SEQUENCE],
            ),
        ]
    )
    ds.DigitalSignaturesSequence = Sequence(
        [
            _signature("1.2.3.1", 1, [PATIENT_NAME, PATIENT_ID]),
            _signature("1.2.3.2", 2),
            _signature("1.2.3.4", 2, [STUDY_DATE]),
        ]
    )
    return ds


def _uids(plan):
    return {inv.uid for inv in plan.invalidated_signatures}


def _mac_references(scope):
    """Return the MAC IDs referenced by signatures and present in a scope."""
    referenced = {
        item.MACIDNumber for item in scope.DigitalSignaturesSequence
    }
    present = {item.MACIDNumber for item in scope.MACParametersSequence}
    return referenced, present


class TestPlanCreation:
    def test_plan_is_bound_to_dataset(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        assert isinstance(plan, SignatureEditPlan)
        assert plan.dataset is ds

    def test_plan_chaining_and_clear(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        assert plan.set("PatientBirthDate", "x") is plan
        assert plan.delete("PatientBirthDate") is plan
        plan.clear()
        plan.apply()
        assert ds.PatientBirthDate == "19800101"


class TestInvalidationListing:
    def test_covered_public_tag_invalidates_one_signature(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientName", "Smith^Jane")
        assert _uids(plan) == {"1.2.3.1"}

    def test_overlapping_coverage_lists_both_signatures(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientID", "999")
        assert _uids(plan) == {"1.2.3.1", "1.2.3.2"}

    def test_uncovered_public_tag_invalidates_nothing(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientBirthDate", "19790101")
        assert plan.invalidated_signatures == []
        plan.apply(policy="reject")
        assert ds.PatientBirthDate == "19790101"

    def test_uncovered_private_tag_invalidates_nothing(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(PRIVATE_TAG, "updated")
        assert plan.invalidated_signatures == []
        plan.apply()
        assert ds[PRIVATE_TAG].value == "updated"
        assert len(ds.DigitalSignaturesSequence) == 3

    def test_new_private_tag_with_vr_invalidates_nothing(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(Tag(0x0011, 0x1001), "x", VR="LO")
        assert plan.invalidated_signatures == []
        plan.apply()
        assert ds[Tag(0x0011, 0x1001)].value == "x"

    def test_new_unknown_tag_without_vr_raises(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(Tag(0x0011, 0x1001), "x")
        with pytest.raises(ValueError, match="VR"):
            plan.apply()

    def test_sequence_subelement_invalidates_covering_and_nested(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(
            ("ReferencedSeriesSequence", 0, "StudyInstanceUID"), "9.9.9"
        )
        invalidated = plan.invalidated_signatures
        assert _uids(plan) == {"1.2.3.2", "1.2.3.3"}
        nested = next(inv for inv in invalidated if inv.uid == "1.2.3.3")
        assert nested.scope_description == "ReferencedSeriesSequence[0]"
        assert nested.tags == {STUDY_INSTANCE_UID}
        top = next(inv for inv in invalidated if inv.uid == "1.2.3.2")
        assert top.scope_description == "top-level dataset"
        assert top.tags == {REFERENCED_SERIES_SEQUENCE}

    def test_change_in_covered_sequence_still_invalidates(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(
            ("ReferencedSeriesSequence", 0, "SeriesInstanceUID"), "9.9.8"
        )
        # SeriesInstanceUID is not signed at the nested scope (sig C covers
        # only StudyInstanceUID), but sig B covers the parent sequence.
        assert _uids(plan) == {"1.2.3.2"}

    def test_change_in_uncovered_sequence_keeps_all_signatures(self):
        ds = make_signed_dataset()
        item = Dataset()
        item.CodeValue = "x"
        ds.ProcedureCodeSequence = Sequence([item])
        plan = ds.signature_edit_plan()
        plan.set(("ProcedureCodeSequence", 0, "CodeValue"), "y")
        assert plan.invalidated_signatures == []
        plan.apply()
        assert item.CodeValue == "y"
        assert len(ds.DigitalSignaturesSequence) == 3

    def test_deleted_covered_tag_invalidates(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.delete("StudyDate")
        assert _uids(plan) == {"1.2.3.2", "1.2.3.4"}

    def test_invalidated_signature_is_dataclass(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientName", "x")
        inv = plan.invalidated_signatures[0]
        assert isinstance(inv, InvalidatedSignature)
        assert inv.scope is ds
        assert inv.index == 0
        assert inv.scope_path == ()
        assert inv.tags == {PATIENT_NAME}


class TestRejectPolicy:
    def test_reject_raises_and_changes_nothing(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientName", "Smith^Jane")
        with pytest.raises(SignatureInvalidationError) as exc:
            plan.apply(policy="reject")
        assert exc.value.invalidated[0].uid == "1.2.3.1"
        assert "1.2.3.1" in str(exc.value)
        assert ds.PatientName == "Doe^John"
        assert len(ds.DigitalSignaturesSequence) == 3
        assert len(ds.MACParametersSequence) == 2

    def test_reject_applies_when_nothing_invalidated(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientBirthDate", "19790101")
        removed = plan.apply(policy="reject")
        assert removed == []
        assert ds.PatientBirthDate == "19790101"

    def test_unknown_policy_raises(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientName", "x")
        with pytest.raises(ValueError, match="policy"):
            plan.apply(policy="nope")


class TestRemovePolicy:
    def test_removes_only_affected_signature_and_orphaned_mac(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientName", "Smith^Jane")
        removed = plan.apply()
        assert [inv.uid for inv in removed] == ["1.2.3.1"]
        assert ds.PatientName == "Smith^Jane"
        remaining = [
            s.DigitalSignatureUID for s in ds.DigitalSignaturesSequence
        ]
        assert remaining == ["1.2.3.2", "1.2.3.4"]
        # MAC 1 is no longer referenced; MAC 2 is shared by B and D
        mac_ids = [m.MACIDNumber for m in ds.MACParametersSequence]
        assert mac_ids == [2]

    def test_shared_mac_parameter_kept_while_other_signature_remains(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("StudyDate", "20250101")
        removed = plan.apply()
        assert sorted(inv.uid for inv in removed) == ["1.2.3.2", "1.2.3.4"]
        remaining = [
            s.DigitalSignatureUID for s in ds.DigitalSignaturesSequence
        ]
        assert remaining == ["1.2.3.1"]
        mac_ids = [m.MACIDNumber for m in ds.MACParametersSequence]
        assert mac_ids == [1]
        assert ds.StudyDate == "20250101"

    def test_remove_last_signature_drops_sequences(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientID", "999")
        plan.set("StudyDate", "20250101")
        plan.apply()
        assert "DigitalSignaturesSequence" not in ds
        assert "MACParametersSequence" not in ds
        assert ds.PatientID == "999"
        assert ds.StudyDate == "20250101"

    def test_sequence_subelement_remove_cleans_both_scopes(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(
            ("ReferencedSeriesSequence", 0, "StudyInstanceUID"), "9.9.9"
        )
        plan.apply()
        nested = ds.ReferencedSeriesSequence[0]
        assert nested.StudyInstanceUID == "9.9.9"
        assert "DigitalSignaturesSequence" not in nested
        assert "MACParametersSequence" not in nested
        remaining = [
            s.DigitalSignatureUID for s in ds.DigitalSignaturesSequence
        ]
        assert remaining == ["1.2.3.1", "1.2.3.4"]
        # MAC 2 remains because sig D still references it
        mac_ids = [m.MACIDNumber for m in ds.MACParametersSequence]
        assert mac_ids == [1, 2]

    def test_delete_covered_element_removes_signature(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.delete("StudyDate")
        plan.apply()
        assert "StudyDate" not in ds
        remaining = [
            s.DigitalSignatureUID for s in ds.DigitalSignaturesSequence
        ]
        assert remaining == ["1.2.3.1"]
        mac_ids = [m.MACIDNumber for m in ds.MACParametersSequence]
        assert mac_ids == [1]

    def test_set_dataelement_is_applied(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        elem = DataElement(PATIENT_BIRTH_DATE, "DA", "19790102")
        plan.set("PatientBirthDate", elem)
        plan.apply()
        assert ds.PatientBirthDate == "19790102"


class TestPathErrors:
    def test_missing_sequence_element_raises(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set((Tag(0x0011, 0x1001), 0, "PatientName"), "x")
        with pytest.raises(ValueError, match="not present"):
            plan.invalidated_signatures

    def test_item_index_out_of_range_raises(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(("ReferencedSeriesSequence", 5, "StudyInstanceUID"), "x")
        with pytest.raises(ValueError, match="out of range"):
            plan.invalidated_signatures

    def test_non_sequence_descent_raises(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(("PatientName", 0, "PatientID"), "x")
        with pytest.raises(ValueError, match="not a sequence"):
            plan.invalidated_signatures

    def test_last_component_must_be_element(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        with pytest.raises(ValueError, match="last path component"):
            plan.set(("ReferencedSeriesSequence", 0), "x")

    def test_delete_missing_element_raises(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.delete("Modality")
        with pytest.raises(KeyError):
            plan.apply()


class TestRoundtrip:
    def _roundtrip(self, ds):
        buffer = BytesIO()
        dcmwrite(buffer, ds, little_endian=True, implicit_vr=False)
        buffer.seek(0)
        return dcmread(buffer, force=True)

    def test_references_consistent_after_write_and_reread(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientID", "999")
        plan.apply()

        rt = self._roundtrip(ds)
        assert rt.PatientID == "999"
        # Only sig D (StudyDate) survives at the top level
        remaining = [
            s.DigitalSignatureUID for s in rt.DigitalSignaturesSequence
        ]
        assert remaining == ["1.2.3.4"]
        referenced, present = _mac_references(rt)
        assert referenced == {2}
        assert present == {2}

        # The nested scope is untouched and still self-consistent
        nested = rt.ReferencedSeriesSequence[0]
        ref_nested, present_nested = _mac_references(nested)
        assert ref_nested == {1}
        assert present_nested == {1}

    def test_no_dangling_mac_references_after_nested_removal(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set(
            ("ReferencedSeriesSequence", 0, "StudyInstanceUID"), "9.9.9"
        )
        plan.apply()

        rt = self._roundtrip(ds)
        nested = rt.ReferencedSeriesSequence[0]
        assert nested.StudyInstanceUID == "9.9.9"
        assert "DigitalSignaturesSequence" not in nested
        assert "MACParametersSequence" not in nested

        referenced, present = _mac_references(rt)
        assert referenced == {1, 2}
        assert present == {1, 2}

    def test_unsigned_dataset_roundtrip_unchanged(self):
        ds = make_signed_dataset()
        plan = ds.signature_edit_plan()
        plan.set("PatientBirthDate", "19790101")
        plan.apply()

        rt = self._roundtrip(ds)
        assert rt.PatientBirthDate == "19790101"
        assert len(rt.DigitalSignaturesSequence) == 3
        referenced, present = _mac_references(rt)
        assert referenced == {1, 2}
        assert present == {1, 2}
