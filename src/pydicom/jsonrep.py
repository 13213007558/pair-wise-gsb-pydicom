# Copyright 2008-2021 pydicom authors. See LICENSE file for details.
"""Methods for converting Datasets and DataElements to/from json"""

import base64
from inspect import signature
from typing import TypeAlias, Any, cast, TYPE_CHECKING
from collections.abc import Callable

from pydicom.misc import warn_and_log
from pydicom.valuerep import FLOAT_VR, INT_VR, VR

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


JSON_VALUE_KEYS = ("Value", "BulkDataURI", "InlineBinary")


def convert_to_python_number(value: Any, vr: str) -> Any:
    """When possible convert numeric-like values to either ints or floats
    based on their value representation.

    Parameters
    ----------
    value : Any
        Value of the data element.
    vr : str
        Value representation of the data element.

    Returns
    -------
    Any

        * If `value` is empty or None the empty value for `vr` is returned.
        * If `vr` is an integer-like VR type then returns ``int`` or
          ``List[int]``
        * If `vr` is a float-like VR type then returns ``float`` or
          ``List[float]``
        * Otherwise returns `value` unchanged

    """
    from pydicom.dataelem import empty_value_for_VR

    if value in (None, ""):
        return empty_value_for_VR(vr)

    number_type: type[int] | type[float] | None = None
    if vr in (INT_VR - {VR.AT}) | {VR.US_SS}:
        number_type = int
    if vr in FLOAT_VR:
        number_type = float

    if number_type is None:
        return value

    if isinstance(value, list | tuple):
        return [
            number_type(v) if v is not None else empty_value_for_VR(vr) for v in value
        ]

    return number_type(value)


OtherValueType = None | str | int | float
PNValueType = None | str | dict[str, str]
SQValueType = dict[str, Any] | None  # Recursive

ValueType: TypeAlias = PNValueType | SQValueType | OtherValueType
InlineBinaryType: TypeAlias = str | list[str]
BulkDataURIType: TypeAlias = str | list[str]

JSONValueType = list[ValueType] | InlineBinaryType | BulkDataURIType

BulkDataType = None | str | int | float | bytes
BulkDataHandlerType = Callable[[str, str, str], BulkDataType] | None


class LazyBulkData:
    """A lazily-resolved bulk data value referenced by a "BulkDataURI".

    Keeps the information needed to retrieve the actual value of a bulk
    data element (the URI, the element's tag and VR and the path of any
    enclosing sequence items) without retrieving it. The value is only
    retrieved from the caller-provided bulk data URI handler when
    :meth:`resolve` is called, e.g. when the element's value is accessed
    or the dataset is written to a DICOM file.

    .. versionadded:: 3.1
    """

    def __init__(
        self,
        tag: str,
        vr: str,
        uri: str,
        handler: Callable[[str, str, str], BulkDataType] | None,
        path: tuple[tuple[str, int], ...] = (),
    ) -> None:
        """Create a new lazy bulk data placeholder.

        Parameters
        ----------
        tag : str
            The tag of the element the bulk data belongs to, in uppercase
            hex format like ``"7FE00010"``.
        vr : str
            The value representation of the element the bulk data belongs to.
        uri : str
            The "BulkDataURI" value used to retrieve the bulk data.
        handler : callable or None
            Callable function that accepts the `tag`, `vr` and "BulkDataURI"
            `uri` and returns the actual value of the corresponding data
            element (e.g. retrieved via DICOMweb WADO-RS). May be ``None``,
            in which case :meth:`resolve` raises an exception.
        path : tuple[tuple[str, int], ...], optional
            The path of enclosing sequence items as ``(tag, item index)``
            tuples, ordered from the outermost sequence inwards. Empty for
            top-level elements.
        """
        self.tag = tag
        self.vr = vr
        self.uri = uri
        self.path = path
        self._handler = handler
        self._is_resolved = False
        self._value: BulkDataType = None

    @property
    def is_resolved(self) -> bool:
        """Return ``True`` if the bulk data has been retrieved."""
        return self._is_resolved

    @property
    def value(self) -> BulkDataType:
        """Return the bulk data value, retrieving it if necessary."""
        return self.resolve()

    def resolve(self) -> BulkDataType:
        """Return the bulk data value, retrieving it if necessary.

        The value is retrieved using the bulk data URI handler on the first
        successful call and cached afterwards. A failed retrieval attempt
        is *not* cached - the exception is raised to the caller and a
        subsequent call will re-attempt the retrieval.

        Returns
        -------
        None, str, int, float or bytes
            The value returned by the bulk data URI handler.

        Raises
        ------
        ValueError
            If no bulk data URI handler is available.
        """
        if not self._is_resolved:
            if self._handler is None:
                raise ValueError(
                    "Unable to retrieve the value of data element "
                    f'"{self.tag}" - no bulk data URI handler available'
                )

            value = self._handler(self.tag, self.vr, self.uri)
            self._value = value
            self._is_resolved = True

        return self._value

    def retry(self) -> BulkDataType:
        """Explicitly re-attempt retrieving the bulk data value.

        Forces a new retrieval attempt using the bulk data URI handler,
        even if a previous attempt succeeded or failed.

        Returns
        -------
        None, str, int, float or bytes
            The value returned by the bulk data URI handler.
        """
        self._is_resolved = False
        self._value = None
        return self.resolve()

    def __repr__(self) -> str:
        state = "resolved" if self._is_resolved else "unresolved"
        return (
            f"<LazyBulkData tag={self.tag} vr={self.vr} "
            f"uri={self.uri!r} path={self.path!r} ({state})>"
        )


class JsonDataElementConverter:
    """Convert from a JSON struct to a :class:`DataElement`.

    References
    ----------

    * :dcm:`Annex F of Part 18 of the DICOM Standard<part18/chapter_F.html>`
    * `JSON to Python object conversion table
      <https://docs.python.org/3/library/json.html#json-to-py-table>`_
    """

    def __init__(
        self,
        dataset_class: type["Dataset"],
        tag: str,
        vr: str,
        value: JSONValueType,
        value_key: str | None,
        bulk_data_uri_handler: (
            BulkDataHandlerType | Callable[[str], BulkDataType] | None
        ) = None,
        lazy_bulk_data: bool = False,
        bulk_data_path: tuple[tuple[str, int], ...] = (),
    ) -> None:
        """Create a new converter instance.

        Parameters
        ----------
        dataset_class : dataset.Dataset derived class
            The class object to use for **SQ** element items.
        tag : str
            The data element's tag in uppercase hex format like ``"7FE00010"``.
        vr : str
            The data element value representation.
        value : str or List[None | str | int | float | dict]
            The attribute value for the JSON object's "Value", "InlineBinary"
            or "BulkDataURI" field. If there's no such attribute then `value`
            will be ``[""]``.
        value_key : str or None
            The attribute name for `value`, should be one of:
            ``{"Value", "InlineBinary", "BulkDataURI"}``. If the element's VM
            is ``0`` and none of the keys are used then will be ``None``.
        bulk_data_uri_handler: callable, optional
            Callable function that accepts either the `tag`, `vr` and the
            "BulkDataURI" `value`, or just the "BulkDataURI" `value` of the
            JSON representation of a data element and returns the actual
            value of that data element (retrieved via DICOMweb WADO-RS). If
            no `bulk_data_uri_handler` is specified (default) then the
            corresponding element will have an "empty" value such as
            ``""``, ``b""`` or ``None`` depending on the
            `vr` (i.e. the Value Multiplicity will be 0).
        lazy_bulk_data : bool, optional
            If ``True``, elements with a "BulkDataURI" get a
            :class:`~pydicom.jsonrep.LazyBulkData` placeholder as their
            value instead of retrieving the bulk data immediately. The
            actual value is only retrieved using the
            `bulk_data_uri_handler` when the element's value is accessed.
            If ``False`` (default) the bulk data is retrieved during
            decoding.
        bulk_data_path : tuple[tuple[str, int], ...], optional
            The path of enclosing sequence items as ``(tag, item index)``
            tuples, used for the :attr:`LazyBulkData.path` of lazily
            decoded bulk data. Empty for top-level elements (default).
        """
        self.dataset_class = dataset_class
        self.tag = tag
        self.vr = vr
        self.value = value
        self.value_key = value_key
        self.bulk_data_element_handler: BulkDataHandlerType
        self.lazy_bulk_data = lazy_bulk_data
        self.bulk_data_path = bulk_data_path

        handler = bulk_data_uri_handler
        if handler and len(signature(handler).parameters) == 1:
            # `handler` is Callable[[str], BulkDataType]
            def wrapper(tag: str, vr: str, value: str) -> BulkDataType:
                x = cast(Callable[[str], BulkDataType], handler)
                return x(value)

            self.bulk_data_element_handler = wrapper
        else:
            self.bulk_data_element_handler = cast(BulkDataHandlerType, handler)

    def get_element_values(self) -> Any:
        """Return a the data element value or list of values.

        Returns
        -------
        None, str, float, int, bytes, dataset_class or a list of these
            The value or value list of the newly created data element.
        """
        from pydicom.dataelem import empty_value_for_VR

        # An attribute with an empty value should have no "Value",
        #   "BulkDataURI" or "InlineBinary"
        if self.value_key is None:
            return empty_value_for_VR(self.vr)

        if self.value_key == "Value":
            if not isinstance(self.value, list):
                raise TypeError(
                    f"'{self.value_key}' of data element '{self.tag}' must be a list"
                )

            if not self.value:
                return empty_value_for_VR(self.vr)

            val = cast(list[ValueType], self.value)
            element_value = [
                self.get_regular_element_value(v, item_index=index)
                for index, v in enumerate(val)
            ]
            if len(element_value) == 1 and self.vr != VR.SQ:
                element_value = element_value[0]

            return convert_to_python_number(element_value, self.vr)

        # The value for "InlineBinary" shall be encoded as a base64 encoded
        # string, as shown in PS3.18, Table F.3.1-1, but the example in
        # PS3.18, Annex F.4 shows the string enclosed in a list.
        # We support both variants, as the standard is ambiguous here,
        # and do the same for "BulkDataURI".
        value = cast(str | list[str], self.value)
        if isinstance(value, list):
            value = value[0]

        if self.value_key == "InlineBinary":
            # The `value` should be a base64 encoded str
            if not isinstance(value, str):
                raise TypeError(
                    f"Invalid attribute value for data element '{self.tag}' - "
                    "the value for 'InlineBinary' must be str, not "
                    f"{type(value).__name__}"
                )

            return base64.b64decode(value)  # bytes

        if self.value_key == "BulkDataURI":
            # The `value` should be a URI as a str
            if not isinstance(value, str):
                raise TypeError(
                    f"Invalid attribute value for data element '{self.tag}' - "
                    "the value for 'BulkDataURI' must be str, not "
                    f"{type(value).__name__}"
                )

            if self.bulk_data_element_handler is None:
                if not self.lazy_bulk_data:
                    warn_and_log(
                        "No bulk data URI handler provided for retrieval "
                        f'of value of data element "{self.tag}"'
                    )
                    return empty_value_for_VR(self.vr)

            if self.lazy_bulk_data:
                return LazyBulkData(
                    tag=self.tag,
                    vr=self.vr,
                    uri=value,
                    handler=self.bulk_data_element_handler,
                    path=self.bulk_data_path,
                )

            return self.bulk_data_element_handler(self.tag, self.vr, value)

        raise ValueError(
            f"Unknown attribute name '{self.value_key}' for tag {self.tag}"
        )

    def get_regular_element_value(
        self, value: ValueType, item_index: int | None = None
    ) -> Any:
        """Return a the data element value created from a json "Value" entry.

        Parameters
        ----------
        value : None, str, int, float or dict
            The data element's value from the json entry.
        item_index : int or None, optional
            For **SQ** elements, the index of the item within the sequence,
            used to build the sequence item path for lazily decoded bulk
            data. Ignored otherwise.

        Returns
        -------
        None, str, int, float or Dataset
            A single value of the corresponding :class:`DataElement`.
        """
        from pydicom.dataelem import empty_value_for_VR

        # Table F.2.3-1 has JSON type mappings
        if self.vr == VR.SQ:  # Dataset
            # May be an empty dict
            value = cast(dict[str, Any], value)
            return self.get_sequence_item(value, item_index)

        if value is None:
            return empty_value_for_VR(self.vr)

        if self.vr == VR.PN:  # str
            value = cast(dict[str, str], value)
            return self.get_pn_element_value(value)

        if self.vr == VR.AT:  # Optional[int]
            # May be an empty str
            value = cast(str, value)
            try:
                return int(value, 16)
            except ValueError:
                warn_and_log(f"Invalid value '{value}' for AT element - ignoring it")

            return None

        return value

    def get_sequence_item(
        self, value: SQValueType, item_index: int | None = None
    ) -> "Dataset":
        """Return a sequence item for the JSON dict `value`.

        Parameters
        ----------
        value : dict or None
            The sequence item from the JSON entry.
        item_index : int or None, optional
            The index of the item within the sequence, used to build the
            sequence item path for lazily decoded bulk data.

        Returns
        -------
        dataset_class
            The decoded dataset item.

        Raises
        ------
        KeyError
            If the "vr" key is missing for a contained element
        """
        from pydicom import DataElement
        from pydicom.dataelem import empty_value_for_VR

        ds = self.dataset_class()

        if item_index is None:
            bulk_data_path = self.bulk_data_path
        else:
            bulk_data_path = self.bulk_data_path + ((self.tag, item_index),)

        value = {} if value is None else value
        for key, val in value.items():
            if "vr" not in val:
                raise KeyError(f"Data element '{self.tag}' must have key 'vr'")

            vr = val["vr"]
            unique_value_keys = tuple(set(val.keys()) & set(JSON_VALUE_KEYS))

            if not unique_value_keys:
                # data element with no value
                elem = DataElement(
                    tag=int(key, 16), value=empty_value_for_VR(vr), VR=vr
                )
            else:
                value_key = unique_value_keys[0]
                elem = DataElement.from_json(
                    self.dataset_class,
                    key,
                    vr,
                    val[value_key],
                    value_key,
                    self.bulk_data_element_handler,
                    lazy_bulk_data=self.lazy_bulk_data,
                    bulk_data_path=bulk_data_path,
                )
            ds.add(elem)

        return ds

    def get_pn_element_value(self, value: str | dict[str, str]) -> str:
        """Return a person name from JSON **PN** value as str.

        Values with VR PN have a special JSON encoding, see the DICOM Standard,
        Part 18, :dcm:`Annex F.2.2<part18/sect_F.2.2.html>`.

        Parameters
        ----------
        value : Dict[str, str]
            The person name components in the JSON entry.

        Returns
        -------
        str
            The decoded PersonName object or an empty string.
        """
        if not isinstance(value, dict):
            # Some DICOMweb services get this wrong, so we
            # workaround the issue and warn the user
            # rather than raising an error.
            warn_and_log(
                f"Value of data element '{self.tag}' with VR Person Name (PN) "
                "is not formatted correctly"
            )
            return value

        if "Phonetic" in value:
            comps = ["", "", ""]
        elif "Ideographic" in value:
            comps = ["", ""]
        else:
            comps = [""]

        if "Alphabetic" in value:
            comps[0] = value["Alphabetic"]
        if "Ideographic" in value:
            comps[1] = value["Ideographic"]
        if "Phonetic" in value:
            comps[2] = value["Phonetic"]

        return "=".join(comps)
