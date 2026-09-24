# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Tests for the pydicom.digisig module."""

from io import BytesIO

import pytest

from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset
from pydicom.digisig import (
    InvalidatedSignature,
    SignatureEditPlan,
    SignatureInvalidationError,
)
from pydicom.sequence import Sequence
from pydicom.tag import Tag

SIG_A_UID = "1.2.840.10008.1"
SIG_B_UID = "1.2.840.10008.2"
SIG_C_UID = "1.2.840.10008.3"

PATIENT_NAME = Tag(0x00100010)
PATIENT_ID = Tag(0x00100020)
STUDY_DATE = Tag(0x00080020)
REF_SERIES_SEQ = Tag(0x00081115)
SERIES_INSTANCE_UID = Tag(0x0020000E)
PRIVATE_TAG = 0x00111001


def make_signature(uid, mac_id, signed_tags):
    """Return a Digital Signatures Sequence item covering `signed_tags`."""
    item = Dataset()
    item.DigitalSignatureUID = uid
    mac = Dataset()
    mac.MACIDNumber = mac_id
    mac.DataElementsSigned = list(signed_tags)
    item.MACParametersSequence = Sequence([mac])
    return item


def make_shared_mac(mac_id):
    mac = Dataset()
    mac.MACIDNumber = mac_id
    return mac


@pytest.fixture
def signed_ds():
    """Dataset with two overlapping top-level signatures and a nested one.

    * Signature A (MAC ID 1) covers PatientName and PatientID
    * Signature B (MAC ID 2) covers PatientID, StudyDate and
      ReferencedSeriesSequence
    * Signature C (MAC ID 3) is nested in ReferencedSeriesSequence item 0
      and covers SeriesInstanceUID

    Shared MAC parameters for MAC IDs 1 and 2 are kept at the top level.
    """
    ds = Dataset()
    ds.PatientName = "Doe^John"
    ds.PatientID = "12345"
    ds.StudyDate = "20240101"
    ds.add_new(PRIVATE_TAG, "LO", "PRIVATE")

    item = Dataset()
    item.SeriesInstanceUID = "1.2.3.9"
    item.Manufacturer = "ACME"
    item.DigitalSignaturesSequence = Sequence(
        [make_signature(SIG_C_UID, 3, [SERIES_INSTANCE_UID])]
    )
    ds.ReferencedSeriesSequence = Sequence([item])

    ds.DigitalSignaturesSequence = Sequence(
        [
            make_signature(SIG_A_UID, 1, [PATIENT_NAME, PATIENT_ID]),
            make_signature(SIG_B_UID, 2, [PATIENT_ID, STUDY_DATE, REF_SERIES_SEQ]),
        ]
    )
    ds.MACParametersSequence = Sequence([make_shared_mac(1), make_shared_mac(2)])
    return ds


def signature_uids(ds):
    seq = ds.get("DigitalSignaturesSequence")
    if seq is None:
        return []
    return [item.DigitalSignatureUID for item in seq]


class TestInvalidationListing:
    def test_change_covered_public_tag(self, signed_ds):
        """A change to a tag covered by overlapping signatures lists both."""
        plan = SignatureEditPlan(signed_ds)
        plan.change("PatientID", "67890")

        invalidated = plan.invalidated_signatures
        assert {sig.signature_uid for sig in invalidated} == {SIG_A_UID, SIG_B_UID}
        for sig in invalidated:
            assert isinstance(sig, InvalidatedSignature)
            assert sig.triggered_by == [PATIENT_ID]
            assert sig.path == ()
            assert sig.scope is signed_ds
        assert {sig.mac_id for sig in invalidated} == {1, 2}
        assert {sig.index for sig in invalidated} == {0, 1}

    def test_change_uncovered_private_tag(self, signed_ds):
        """A change to an uncovered private tag invalidates nothing."""
        plan = signed_ds.plan_signature_edits()
        plan.change(PRIVATE_TAG, "UPDATED")

        assert plan.invalidated_signatures == []
        plan.commit()
        assert signed_ds[PRIVATE_TAG].value == "UPDATED"
        assert signature_uids(signed_ds) == [SIG_A_UID, SIG_B_UID]
        # MAC parameters are untouched when no signature is removed
        assert [mac.MACIDNumber for mac in signed_ds.MACParametersSequence] == [1, 2]

    def test_change_nested_sequence_child(self, signed_ds):
        """A change inside a signed sequence invalidates nested and parent
        signatures covering the sequence element."""
        item = signed_ds.ReferencedSeriesSequence[0]
        plan = SignatureEditPlan(signed_ds)
        plan.change(item, "SeriesInstanceUID", "9.8.7.6")

        invalidated = plan.invalidated_signatures
        by_uid = {sig.signature_uid: sig for sig in invalidated}
        assert set(by_uid) == {SIG_B_UID, SIG_C_UID}

        nested = by_uid[SIG_C_UID]
        assert nested.path == ((REF_SERIES_SEQ, 0),)
        assert nested.scope is item
        assert nested.triggered_by == [SERIES_INSTANCE_UID]

        parent = by_uid[SIG_B_UID]
        assert parent.path == ()
        assert parent.triggered_by == [REF_SERIES_SEQ]

    def test_change_uncovered_nested_child(self, signed_ds):
        """A nested change to an uncovered element only propagates to
        ancestors covering the sequence element."""
        item = signed_ds.ReferencedSeriesSequence[0]
        plan = SignatureEditPlan(signed_ds)
        plan.change(item, "Manufacturer", "OTHER")

        invalidated = plan.invalidated_signatures
        assert [sig.signature_uid for sig in invalidated] == [SIG_B_UID]

    def test_no_signatures_no_invalidation(self):
        ds = Dataset()
        ds.PatientName = "Doe^John"
        plan = SignatureEditPlan(ds)
        plan.change("PatientName", "Doe^Jane")
        assert plan.invalidated_signatures == []
        plan.commit()
        assert ds.PatientName == "Doe^Jane"

    def test_change_unreachable_dataset_raises(self, signed_ds):
        plan = SignatureEditPlan(signed_ds)
        with pytest.raises(ValueError, match="not reachable"):
            plan.change(Dataset(), "PatientName", "X")

    def test_two_level_nesting(self, signed_ds):
        """Changes propagate through multiple nesting levels, marking each
        enclosing sequence element at its own scope."""
        inner = Dataset()
        inner.PatientName = "Inner^Name"
        inner.DigitalSignaturesSequence = Sequence(
            [make_signature("1.2.840.10008.4", 4, [PATIENT_NAME])]
        )
        item = signed_ds.ReferencedSeriesSequence[0]
        item.SourceImageSequence = Sequence([inner])

        plan = SignatureEditPlan(signed_ds)
        plan.change(inner, "PatientName", "Changed^Name")

        invalidated = plan.invalidated_signatures
        by_uid = {sig.signature_uid: sig for sig in invalidated}
        # Inner signature, plus signature B covering ReferencedSeriesSequence
        assert set(by_uid) == {"1.2.840.10008.4", SIG_B_UID}
        assert by_uid["1.2.840.10008.4"].path == (
            (REF_SERIES_SEQ, 0),
            (Tag(0x00082112), 0),
        )

        plan.commit("remove")
        assert inner.PatientName == "Changed^Name"
        assert "DigitalSignaturesSequence" not in inner
        assert signature_uids(signed_ds) == [SIG_A_UID]

    def test_change_invalid_argument_count(self, signed_ds):
        plan = SignatureEditPlan(signed_ds)
        with pytest.raises(TypeError):
            plan.change("PatientName")


class TestCommit:
    def test_commit_raise_rejects_edit(self, signed_ds):
        """The default mode rejects edits that would invalidate signatures."""
        plan = SignatureEditPlan(signed_ds)
        plan.change("PatientName", "Doe^Jane")

        with pytest.raises(SignatureInvalidationError) as exc_info:
            plan.commit()

        assert [s.signature_uid for s in exc_info.value.invalidated] == [SIG_A_UID]
        assert SIG_A_UID in str(exc_info.value)
        # Nothing was applied and the plan is still pending
        assert signed_ds.PatientName == "Doe^John"
        assert plan.pending

        plan.discard()
        assert not plan.pending

    def test_commit_raise_no_invalidation_applies(self, signed_ds):
        plan = SignatureEditPlan(signed_ds)
        plan.change(PRIVATE_TAG, "UPDATED")
        assert plan.commit() == []
        assert signed_ds[PRIVATE_TAG].value == "UPDATED"
        assert not plan.pending

    def test_commit_invalid_mode(self, signed_ds):
        plan = SignatureEditPlan(signed_ds)
        plan.change(PRIVATE_TAG, "UPDATED")
        with pytest.raises(ValueError, match="on_invalidated"):
            plan.commit("bogus")

    def test_commit_remove_common_tag(self, signed_ds):
        """Removing overlapping signatures clears the sequence and the
        now-unreferenced shared MAC parameters."""
        plan = SignatureEditPlan(signed_ds)
        plan.change("PatientID", "67890")
        invalidated = plan.commit("remove")

        assert {sig.signature_uid for sig in invalidated} == {SIG_A_UID, SIG_B_UID}
        assert signed_ds.PatientID == "67890"
        # Both top-level signatures were removed, element is gone entirely
        assert "DigitalSignaturesSequence" not in signed_ds
        # No remaining signature references MAC ID 1 or 2
        assert "MACParametersSequence" not in signed_ds
        # The nested signature is unaffected
        item = signed_ds.ReferencedSeriesSequence[0]
        assert signature_uids(item) == [SIG_C_UID]

    def test_commit_remove_keeps_referenced_mac(self, signed_ds):
        """Only the affected signature is removed; MAC parameters still
        referenced by a remaining signature are kept."""
        plan = SignatureEditPlan(signed_ds)
        plan.change("StudyDate", "20240202")
        invalidated = plan.commit("remove")

        assert [sig.signature_uid for sig in invalidated] == [SIG_B_UID]
        assert signed_ds.StudyDate == "20240202"
        assert signature_uids(signed_ds) == [SIG_A_UID]
        # MAC ID 1 is still referenced by signature A, MAC ID 2 is pruned
        assert [mac.MACIDNumber for mac in signed_ds.MACParametersSequence] == [1]
        # The remaining signature keeps its own MAC parameters
        remaining = signed_ds.DigitalSignaturesSequence[0]
        assert remaining.MACParametersSequence[0].MACIDNumber == 1

    def test_commit_remove_nested(self, signed_ds):
        """Nested scopes are cleaned up independently."""
        item = signed_ds.ReferencedSeriesSequence[0]
        plan = SignatureEditPlan(signed_ds)
        plan.change(item, "SeriesInstanceUID", "9.8.7.6")
        plan.commit("remove")

        assert item.SeriesInstanceUID == "9.8.7.6"
        # Nested signature sequence is empty and removed
        assert "DigitalSignaturesSequence" not in item
        # Top-level signature B was covering the sequence, A remains
        assert signature_uids(signed_ds) == [SIG_A_UID]
        assert [mac.MACIDNumber for mac in signed_ds.MACParametersSequence] == [1]

    def test_commit_remove_applies_data_element(self, signed_ds):
        from pydicom.dataelem import DataElement

        plan = SignatureEditPlan(signed_ds)
        plan.change("PatientName", DataElement(0x00100010, "PN", "Doe^Jane"))
        with pytest.raises(SignatureInvalidationError):
            plan.commit()
        plan.commit("remove")
        assert signed_ds.PatientName == "Doe^Jane"


class TestRoundTrip:
    def test_save_and_reread_references(self, signed_ds):
        """After removing an invalidated signature, the saved dataset keeps
        consistent signature/MAC references when re-read."""
        plan = SignatureEditPlan(signed_ds)
        plan.change("StudyDate", "20240202")
        plan.commit("remove")

        buffer = BytesIO()
        dcmwrite(buffer, signed_ds, implicit_vr=False, little_endian=True)
        buffer.seek(0)
        reread = dcmread(buffer, force=True)

        # Only signature A remains, with its MAC parameters intact
        assert signature_uids(reread) == [SIG_A_UID]
        remaining = reread.DigitalSignaturesSequence[0]
        assert remaining.MACParametersSequence[0].MACIDNumber == 1
        assert list(remaining.MACParametersSequence[0].DataElementsSigned) == [
            PATIENT_NAME,
            PATIENT_ID,
        ]
        # No dangling MAC parameter references
        assert [mac.MACIDNumber for mac in reread.MACParametersSequence] == [1]
        # The nested signature survived the round trip
        item = reread.ReferencedSeriesSequence[0]
        assert signature_uids(item) == [SIG_C_UID]
        assert (
            item.DigitalSignaturesSequence[0]
            .MACParametersSequence[0]
            .DataElementsSigned
            == SERIES_INSTANCE_UID
        )
        assert reread.StudyDate == "20240202"


class TestHelpers:
    def test_repr(self, signed_ds):
        plan = SignatureEditPlan(signed_ds)
        plan.change("PatientID", "67890")
        text = repr(plan.invalidated_signatures[0])
        assert SIG_A_UID in text
        assert "root dataset" in text

    def test_plan_properties(self, signed_ds):
        plan = signed_ds.plan_signature_edits()
        assert plan.dataset is signed_ds
        assert not plan.pending
        plan.change("PatientID", "67890")
        assert plan.pending
