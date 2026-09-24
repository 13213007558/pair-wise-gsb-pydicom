"""Classification of DICOM UI attributes.

Only tags in INSTANCE_UID_TAGS are remapped. Tags in DEFINITION_UID_TAGS
identify classes, transfer syntaxes, coding schemes, devices or
implementations and are never touched. Any other element with VR UI
(typically private tags) is reported as unclassified and left unchanged
until the caller supplies a rule for it.
"""

from __future__ import annotations

# Instance-identity UIDs: these get replaced.
INSTANCE_UID_TAGS: dict[tuple[int, int], str] = {
    (0x0002, 0x0003): "MediaStorageSOPInstanceUID",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x0058): "FailedSOPInstanceUIDList",
    (0x0008, 0x1155): "ReferencedSOPInstanceUID",
    (0x0008, 0x1195): "TransactionUID",
    (0x0008, 0x3010): "IrradiationEventUID",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0052): "FrameOfReferenceUID",
    (0x0020, 0x0200): "SynchronizationFrameOfReferenceUID",
    (0x0020, 0x9161): "ConcatenationUID",
    (0x0020, 0x9164): "DimensionOrganizationUID",
    (0x0040, 0xA124): "UID",
    (0x3006, 0x0024): "ReferencedFrameOfReferenceUID",
}

# Definition-class UIDs: classes, transfer syntaxes, coding schemes,
# devices, implementations. Never remapped.
DEFINITION_UID_TAGS: dict[tuple[int, int], str] = {
    (0x0002, 0x0002): "MediaStorageSOPClassUID",
    (0x0002, 0x0010): "TransferSyntaxUID",
    (0x0002, 0x0012): "ImplementationClassUID",
    (0x0008, 0x0014): "InstanceCreatorUID",
    (0x0008, 0x0016): "SOPClassUID",
    (0x0008, 0x010C): "CodingSchemeUID",
    (0x0008, 0x0117): "ContextUID",
    (0x0008, 0x1150): "ReferencedSOPClassUID",
    (0x0018, 0x1002): "DeviceUID",
    (0x0040, 0xDB0C): "TemplateExtensionOrganizationUID",
    (0x0040, 0xDB0D): "TemplateExtensionCreatorUID",
}


def classify(tag: tuple[int, int]) -> str:
    """Return 'instance', 'definition' or 'unknown' for a UI element tag."""
    if tag in INSTANCE_UID_TAGS:
        return "instance"
    if tag in DEFINITION_UID_TAGS:
        return "definition"
    return "unknown"

