# Copyright 2008-2019 pydicom authors. See LICENSE file for details.
import base64
import copy
import json
import logging
from unittest import mock

import pytest

from pydicom import dcmread
from pydicom.config import have_numpy
from pydicom.data import get_testdata_file
from pydicom.dataelem import DataElement, RawDataElement
from pydicom.dataset import Dataset
from pydicom.jsonrep import BulkDataValue
from pydicom.tag import Tag, BaseTag
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.valuerep import PersonName


class TestPersonName:
    def test_json_pn_from_file(self, disable_value_validation):
        with open(get_testdata_file("test_PN.json")) as s:
            ds = Dataset.from_json(s.read())
        assert isinstance(ds[0x00080090].value, PersonName)
        assert isinstance(ds[0x00100010].value, PersonName)
        inner_seq = ds[0x04000561].value[0][0x04000550]
        dataelem = inner_seq[0][0x00100010]
        assert isinstance(dataelem.value, PersonName)

    def test_pn_components_to_json(self):
        def check_name(tag, components):
            # we cannot directly compare the dictionaries, as they are not
            # guaranteed insertion-ordered in Python < 3.7
            value = ds_json[tag]["Value"]
            assert 1 == len(value)
            value = value[0]
            if len(components) == 3:
                assert components[2] == value["Phonetic"]
            else:
                assert "Phonetic" not in value
            if len(components) >= 2:
                assert components[1] == value["Ideographic"]
            else:
                assert "Ideographic" not in value
            assert components[0] == value["Alphabetic"]

        ds = Dataset()
        ds.add_new(0x00100010, "PN", "Yamada^Tarou=山田^太郎=やまだ^たろう")
        ds.add_new(0x00091001, "PN", "Yamada^Tarou")
        ds.add_new(0x00091002, "PN", "Yamada^Tarou==")
        ds.add_new(0x00091003, "PN", "=山田^太郎=やまだ^たろう")
        ds.add_new(0x00091004, "PN", "Yamada^Tarou==やまだ^たろう")
        ds.add_new(0x00091005, "PN", "==やまだ^たろう")
        ds.add_new(0x00091006, "PN", "=山田^太郎")
        ds.add_new(0x00091007, "PN", "Yamada^Tarou=山田^太郎")
        ds.add_new(0x00091008, "PN", None)
        ds.add_new(0x00091009, "PN", "")
        ds.add_new(0x00091010, "PN", [])
        ds_json = ds.to_json_dict()
        check_name("00100010", ["Yamada^Tarou", "山田^太郎", "やまだ^たろう"])
        check_name("00091001", ["Yamada^Tarou"])
        check_name("00091002", ["Yamada^Tarou"])
        check_name("00091003", ["", "山田^太郎", "やまだ^たろう"])
        check_name("00091004", ["Yamada^Tarou", "", "やまだ^たろう"])
        check_name("00091005", ["", "", "やまだ^たろう"])
        check_name("00091006", ["", "山田^太郎"])
        check_name("00091007", ["Yamada^Tarou", "山田^太郎"])
        assert "Value" not in ds_json["00091008"]
        assert "Value" not in ds_json["00091009"]
        assert "Value" not in ds_json["00091010"]

    def test_pn_components_from_json(self):
        # this is the encoded dataset from the previous test, with some
        # empty components omitted
        ds_json = (
            '{"00100010": {"vr": "PN", "Value": [{"Alphabetic": '
            '"Yamada^Tarou", "Ideographic": "山田^太郎", '
            '"Phonetic": "やまだ^たろう"}]}, '
            '"00091001": {"vr": "PN", "Value": '
            '[{"Alphabetic": "Yamada^Tarou"}]}, '
            '"00091002": {"vr": "PN", "Value": '
            '[{"Alphabetic": "Yamada^Tarou", "Ideographic": "", '
            '"Phonetic": ""}]}, '
            '"00091003": {"vr": "PN", "Value": [{'
            '"Ideographic": "山田^太郎", '
            '"Phonetic": "やまだ^たろう"}]}, '
            '"00091004": {"vr": "PN", "Value": '
            '[{"Alphabetic": "Yamada^Tarou", '
            '"Phonetic": "やまだ^たろう"}]}, '
            '"00091005": {"vr": "PN", "Value": '
            '[{"Phonetic": "やまだ^たろう"}]}, '
            '"00091006": {"vr": "PN", "Value":'
            ' [{"Ideographic": "山田^太郎"}]}, '
            '"00091007": {"vr": "PN", "Value": '
            '[{"Alphabetic": "Yamada^Tarou", '
            '"Ideographic": "山田^太郎"}]}}'
        )

        ds = Dataset.from_json(ds_json)
        assert "Yamada^Tarou=山田^太郎=やまだ^たろう" == ds.PatientName
        assert "Yamada^Tarou" == ds[0x00091001].value
        assert "Yamada^Tarou" == ds[0x00091002].value
        assert "=山田^太郎=やまだ^たろう" == ds[0x00091003].value
        assert "Yamada^Tarou==やまだ^たろう" == ds[0x00091004].value
        assert "==やまだ^たろう" == ds[0x00091005].value
        assert "=山田^太郎" == ds[0x00091006].value
        assert "Yamada^Tarou=山田^太郎" == ds[0x00091007].value

    def test_empty_value(self):
        ds = Dataset()
        ds.add_new(0x00100010, "PN", "")
        ds_json = ds.to_json_dict()
        assert "00100010" in ds_json
        assert "Value" not in ds_json["00100010"]

    def test_multi_value_to_json(self):
        ds = Dataset()
        patient_names = ["Buc^Jérôme", "Διονυσιος", "Люкceмбypг"]
        ds.add_new(0x00091001, "PN", patient_names)
        ds_json = ds.to_json_dict()
        assert [
            {"Alphabetic": "Buc^Jérôme"},
            {"Alphabetic": "Διονυσιος"},
            {"Alphabetic": "Люкceмбypг"},
        ] == ds_json["00091001"]["Value"]

    def test_dataelem_from_json(self):
        tag = 0x0080090
        vr = "PN"
        value = [{"Alphabetic": ""}]
        dataelem = DataElement.from_json(Dataset, tag, vr, value, "Value")
        assert isinstance(dataelem.value, PersonName)


class TestAT:
    def test_to_json(self):
        ds = Dataset()
        ds.add_new(0x00091001, "AT", [0x00100010, 0x00100020])
        ds.add_new(0x00091002, "AT", Tag(0x28, 0x02))
        ds.add_new(0x00091003, "AT", BaseTag(0x00280002))
        ds.add_new(0x00091004, "AT", [0x00280002, Tag("PatientName")])
        ds.add_new(0x00091005, "AT", None)
        ds.add_new(0x00091006, "AT", "")
        ds.add_new(0x00091007, "AT", [])
        ds_json = ds.to_json_dict()

        assert ["00100010", "00100020"] == ds_json["00091001"]["Value"]
        assert ["00280002"] == ds_json["00091002"]["Value"]
        assert ["00280002"] == ds_json["00091003"]["Value"]
        assert ["00280002", "00100010"] == ds_json["00091004"]["Value"]
        assert "Value" not in ds_json["00091005"]
        assert "Value" not in ds_json["00091006"]
        assert "Value" not in ds_json["00091007"]

    def test_from_json(self):
        ds_json = (
            '{"00091001": {"vr": "AT", "Value": ["000910AF"]}, '
            '"00091002": {"vr": "AT", "Value": ["00100010", '
            '"00100020", "00100030"]}}'
        )
        ds = Dataset.from_json(ds_json)
        assert 0x000910AF == ds[0x00091001].value
        assert [0x00100010, 0x00100020, 0x00100030] == ds[0x00091002].value

    def test_invalid_value_in_json(self):
        ds_json = (
            '{"00091001": {"vr": "AT", "Value": ["000910AG"]}, '
            '"00091002": {"vr": "AT", "Value": ["00100010"]}}'
        )
        with pytest.warns(
            UserWarning,
            match="Invalid value '000910AG' for AT element - ignoring it",
        ):
            ds = Dataset.from_json(ds_json)
            assert ds[0x00091001].value is None
            assert 0x00100010 == ds[0x00091002].value

    def test_invalid_tag_in_json(self):
        ds_json = (
            '{"000910AG": {"vr": "AT", "Value": ["00091000"]}, '
            '"00091002": {"vr": "AT", "Value": ["00100010"]}}'
        )
        with pytest.raises(
            ValueError,
            match="Data element '000910AG' could not be loaded from JSON:",
        ):
            ds = Dataset.from_json(ds_json)
            assert ds[0x00091001].value is None
            assert 0x00100010 == ds[0x00091002].value


class TestDataSetToJson:
    def test_json_from_dicom_file(self, no_numpy_use, disable_value_validation):
        ds1 = dcmread(get_testdata_file("CT_small.dcm"))
        ds_json = ds1.to_json()
        ds2 = Dataset.from_json(ds_json)
        assert ds1 == ds2

        ds_json = ds1.to_json_dict()
        ds2 = Dataset.from_json(ds_json)
        assert ds1 == ds2

    def test_roundtrip(self):
        ds = Dataset()
        ds.add_new(0x00080005, "CS", "ISO_IR 100")
        ds.add_new(0x00090010, "LO", "Creator 1.0")
        ds.add_new(0x00091001, "SH", "Version1")
        ds.add_new(0x00091002, "OB", b"BinaryContent")
        ds.add_new(0x00091003, "OW", b"\x0102\x3040\x5060")
        ds.add_new(0x00091004, "OF", b"\x00\x01\x02\x03\x04\x05\x06\x07")
        ds.add_new(
            0x00091005,
            "OD",
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x01\x01\x02\x03\x04\x05\x06\x07",
        )
        ds.add_new(
            0x00091006, "OL", b"\x00\x01\x02\x03\x04\x05\x06\x07\x01\x01\x02\x03"
        )
        ds.add_new(0x00091007, "UI", "1.2.3.4.5.6")
        ds.add_new(0x00091008, "DA", "20200101")
        ds.add_new(0x00091009, "TM", "115500")
        ds.add_new(0x0009100A, "DT", "20200101115500.000000")
        ds.add_new(0x0009100B, "UL", 3000000000)
        ds.add_new(0x0009100C, "SL", -2000000000)
        ds.add_new(0x0009100D, "US", 40000)
        ds.add_new(0x0009100E, "SS", -22222)
        ds.add_new(0x0009100F, "FL", 3.14)
        ds.add_new(0x00091010, "FD", 3.14159265)
        ds.add_new(0x00091011, "CS", "TEST MODE")
        ds.add_new(0x00091012, "PN", "CITIZEN^1")
        ds.add_new(0x00091013, "PN", "Yamada^Tarou=山田^太郎=やまだ^たろう")
        ds.add_new(0x00091014, "IS", "42")
        ds.add_new(0x00091015, "DS", "3.14159265")
        ds.add_new(0x00091016, "AE", b"CONQUESTSRV1")
        ds.add_new(0x00091017, "AS", "055Y")
        ds.add_new(0x00091018, "LT", 50 * "Калинка,")
        ds.add_new(0x00091019, "UC", "LONG CODE VALUE")
        ds.add_new(0x0009101A, "UN", b"\x0102\x3040\x5060")
        ds.add_new(0x0009101B, "UR", "https://example.com")
        ds.add_new(0x0009101C, "AT", [0x00100010, 0x00100020])
        ds.add_new(0x0009101D, "ST", 100 * "علي بابا")
        ds.add_new(0x0009101E, "SH", "Διονυσιος")
        ds.add_new(0x00090011, "LO", "Creator 2.0")
        ds.add_new(0x00091101, "SH", "Version2")
        ds.add_new(0x00091102, "US", 2)

        json_string = ds.to_json()
        json_model = json.loads(json_string)

        assert json_model["00080005"]["Value"] == ["ISO_IR 100"]
        assert json_model["00091007"]["Value"] == ["1.2.3.4.5.6"]
        assert json_model["0009100A"]["Value"] == ["20200101115500.000000"]
        assert json_model["0009100B"]["Value"] == [3000000000]
        assert json_model["0009100C"]["Value"] == [-2000000000]
        assert json_model["0009100D"]["Value"] == [40000]
        assert json_model["0009100F"]["Value"] == [3.14]
        assert json_model["00091010"]["Value"] == [3.14159265]
        assert json_model["00091018"]["Value"] == [50 * "Калинка,"]

        ds2 = Dataset.from_json(json_string)
        assert ds == ds2
        ds2 = Dataset.from_json(json_model)
        assert ds == ds2

        json_model2 = ds.to_json_dict()

        assert json_model == json_model2

    def test_dataset_dumphandler(self):
        ds = Dataset()
        ds.add_new(0x00100010, "PN", "Jane^Doe")
        # as the order of the keys is not defined, we have to check both
        assert ds.to_json() in (
            '{"00100010": {"vr": "PN", "Value": [{"Alphabetic": "Jane^Doe"}]}}',
            '{"00100010": {"Value": [{"Alphabetic": "Jane^Doe"}], "vr": "PN"}}',
        )

        assert {
            "00100010": {"vr": "PN", "Value": [{"Alphabetic": "Jane^Doe"}]}
        } == ds.to_json(dump_handler=lambda d: d)

    def test_dataelement_dumphandler(self):
        element = DataElement(0x00100010, "PN", "Jane^Doe")
        # as the order of the keys is not defined, we have to check both
        assert element.to_json() in (
            '{"vr": "PN", "Value": [{"Alphabetic": "Jane^Doe"}]}',
            '{"Value": [{"Alphabetic": "Jane^Doe"}], "vr": "PN"}',
        )

        assert {"vr": "PN", "Value": [{"Alphabetic": "Jane^Doe"}]} == element.to_json(
            dump_handler=lambda d: d
        )

    def test_sort_order(self):
        """Test that tags are serialized in ascending order."""
        ds = Dataset()
        ds.add_new(0x00100040, "CS", "F")
        ds.add_new(0x00100030, "DA", "20000101")
        ds.add_new(0x00100020, "LO", "0017")
        ds.add_new(0x00100010, "PN", "Jane^Doe")

        ds_json = ds.to_json()
        assert ds_json.index('"00100010"') < ds_json.index('"00100020"')
        assert ds_json.index('"00100020"') < ds_json.index('"00100030"')
        assert ds_json.index('"00100030"') < ds_json.index('"00100040"')

    @mock.patch("pydicom.DataElement.to_json_dict", side_effect=ValueError)
    def test_suppress_invalid_tags(self, _):
        """Test tags that raise exceptions don't if suppress_invalid_tags True."""
        ds = Dataset()
        ds.add_new(0x00100010, "PN", "Jane^Doe")

        with pytest.raises(ValueError):
            ds.to_json_dict()

        ds_json = ds.to_json_dict(suppress_invalid_tags=True)

        assert "00100010" not in ds_json

    def test_suppress_invalid_tags_with_failed_dataelement(self, caplog):
        """Test tags that raise exceptions don't if suppress_invalid_tags True."""
        ds = Dataset()
        # we have to add a RawDataElement as creating a DataElement would
        # already raise an exception
        ds[0x00082128] = RawDataElement(
            Tag(0x00082128), "IS", 4, b"5.25", 0, True, True
        )

        with caplog.at_level(logging.WARNING, logger="pydicom"):
            ds_json = ds.to_json_dict(suppress_invalid_tags=True)
            assert "00082128" not in ds_json

        assert (
            "Error while processing tag 00082128: Invalid value for VR IS: '5.25'"
        ) in caplog.text


class TestEmptyJsonValues:
    @pytest.fixture
    def test_json(self):
        yield Dataset.from_json(
            """{
            "00091000": { "vr": "CS", "Value": [ "" ] },
            "00091001": { "vr": "CS", "Value": [ null ] },
            "00091002": { "vr": "LO", "Value": [ "" ] },
            "00091003": { "vr": "LO", "Value": [ null ] },
            "00091004": { "vr": "LT", "Value": [ "" ] },
            "00091005": { "vr": "LT", "Value": [ null ] },
            "00091006": { "vr": "UI", "Value": [ "" ] },
            "00091007": { "vr": "UI", "Value": [ null ] },
            "00091008": { "vr": "DA", "Value": [ "" ] },
            "00091009": { "vr": "DA", "Value": [ null ] },
            "00091020": { "vr": "DS", "Value": [ "" ] },
            "00091021": { "vr": "DS", "Value": [ null ] },
            "00091022": { "vr": "US", "Value": [ "" ] },
            "00091023": { "vr": "US", "Value": [ null ] },
            "00091024": { "vr": "FL", "Value": [ "" ] },
            "00091025": { "vr": "FL", "Value": [ null ] }
        }"""
        )

    def empty_data_from_json_with_none(
        self, test_json, use_none_as_empty_text_VR_value
    ):
        for offset in range(10):
            assert test_json[0x00091000 + offset].value is None

        for offset in range(20, 16):
            assert test_json[0x00091000 + offset].value is None

    def empty_data_from_json_with_str(self, test_json, use_str_as_empty_text_VR_value):
        for offset in range(10):
            assert test_json[0x00091000 + offset].value == ""

        # for non-text VRs (including text VRs converted to other types like DS)
        # None is used as the empty value regardless of the config setting
        for offset in range(20, 16):
            assert test_json[0x00091000 + offset].value is None


class TestSequence:
    def test_nested_sequences(self, disable_value_validation):
        test1_json = get_testdata_file("test1.json")
        with open(test1_json) as f:
            with pytest.warns(UserWarning, match="No bulk data URI handler provided "):
                ds = Dataset.from_json(f.read())
            del ds.PixelData
            assert ds.AcquisitionMatrix == [128, 0, 0, 32, None]

        ds2 = Dataset.from_json(ds.to_json())
        assert ds == ds2


class TestBinary:
    def test_inline_binary(self):
        ds = Dataset()
        ds.add_new(0x00091002, "OB", b"BinaryContent")
        ds_json = ds.to_json_dict()
        assert "00091002" in ds_json
        assert "QmluYXJ5Q29udGVudA==" == ds_json["00091002"]["InlineBinary"]
        ds1 = Dataset.from_json(ds_json)
        assert ds == ds1
        # also test if the binary is enclosed in a list
        ds_json["00091002"]["InlineBinary"] = ["QmluYXJ5Q29udGVudA=="]
        ds1 = Dataset.from_json(ds_json)
        assert ds == ds1

    def test_inline_binary_un_sq(self):
        # Test unknown VR with inline binary and a sequence
        seq_item = Dataset()
        seq_item.update(
            {
                "PatientPosition": "HFS",
                "PatientSetupNumber": "1",
                "SetupTechniqueDescription": "",
            }
        )
        ds_ref = Dataset()
        ds_ref.PatientSetupSequence = [seq_item]

        ds_json = {
            "300A0180": {
                "vr": "UN",
                "InlineBinary": "/v8A4B4AAAAYAABRBAAAAEhGUyAKMIIBAgAAADEgCjCyAQAAAAA=",
            }
        }
        ds1 = Dataset.from_json(ds_json)
        assert "PatientSetupSequence" in ds1
        assert ds1 == ds_ref

    def test_inline_binary_un_pad(self):
        # Test unknown VR with inline binary and odd-length values
        # The value ("HFS") is an odd length.
        # The value is padded with a space before base64 encoding because
        # UN uses implicit VR little endian.
        # When reading back, the padding should be removed.
        ds_json = {
            "00185100": {
                "vr": "UN",
                "InlineBinary": "SEZTIA==",
            }
        }
        ds1 = Dataset.from_json(ds_json)
        assert "PatientPosition" in ds1
        assert ds1.PatientPosition == "HFS"

    def test_invalid_inline_binary(self):
        msg = (
            "Invalid attribute value for data element '00091002' - the value "
            "for 'InlineBinary' must be str, not int"
        )
        ds_json = '{"00091002": {"vr": "OB", "InlineBinary": 42}}'
        with pytest.raises(TypeError, match=msg):
            Dataset.from_json(ds_json)

        ds_json = '{"00091002": {"vr": "OB", "InlineBinary": [42]}}'
        with pytest.raises(TypeError, match=msg):
            Dataset.from_json(ds_json)

    def test_valid_bulkdata_uri(self):
        ds_json = (
            '{"00091002": {"vr": "OB", "BulkDataURI": '
            '"http://example.com/bulkdatahandler"}}'
        )
        msg = r"No bulk data URI handler provided"
        with pytest.warns(UserWarning, match=msg):
            ds = Dataset.from_json(ds_json)
        assert 0x00091002 in ds

        ds_json = (
            '{"00091002": {"vr": "OB", "BulkDataURI": '
            '["http://example.com/bulkdatahandler"]}}'
        )
        with pytest.warns(UserWarning, match=msg):
            ds = Dataset.from_json(ds_json)
        assert 0x00091002 in ds

    def test_invalid_bulkdata_uri(self):
        msg = (
            "Invalid attribute value for data element '00091002' - the value "
            "for 'BulkDataURI' must be str, not int"
        )
        ds_json = '{"00091002": {"vr": "OB", "BulkDataURI": 42}}'
        with pytest.raises(TypeError, match=msg):
            Dataset.from_json(ds_json)

        ds_json = '{"00091002": {"vr": "OB", "BulkDataURI": [42]}}'
        with pytest.raises(TypeError, match=msg):
            Dataset.from_json(ds_json)

    def test_bulk_data_reader_is_called(self):
        def bulk_data_reader(value):
            return b"xyzzy"

        json_data = {"00091002": {"vr": "OB", "BulkDataURI": "https://a.dummy.url"}}
        ds = Dataset().from_json(json.dumps(json_data), bulk_data_reader)

        assert b"xyzzy" == ds[0x00091002].value

    def test_bulk_data_reader_is_called_2(self):
        def bulk_data_reader(tag, vr, value):
            return b"xyzzy"

        json_data = {"00091002": {"vr": "OB", "BulkDataURI": "https://a.dummy.url"}}
        ds = Dataset().from_json(json.dumps(json_data), bulk_data_reader)

        assert b"xyzzy" == ds[0x00091002].value

    def test_bulk_data_reader_is_called_within_SQ(self):
        def bulk_data_reader(_):
            return b"xyzzy"

        json_data = {
            "003a0200": {
                "vr": "SQ",
                "Value": [
                    {"54001010": {"vr": "OW", "BulkDataURI": "https://a.dummy.url"}}
                ],
            }
        }

        ds = Dataset().from_json(json.dumps(json_data), bulk_data_reader)

        assert b"xyzzy" == ds[0x003A0200].value[0][0x54001010].value


class TestNumeric:
    def test_numeric_values(self):
        ds = Dataset()

        ds.add_new(0x0009100B, "UL", 3000000000)
        ds.add_new(0x0009100C, "SL", -2000000000)
        ds.add_new(0x0009100D, "US", 40000)
        ds.add_new(0x0009100E, "SS", -22222)
        ds.add_new(0x0009100F, "FL", 3.14)
        ds.add_new(0x00091010, "FD", 3.14159265)
        ds.add_new(0x00091014, "IS", "42")
        ds.add_new(0x00091015, "DS", "3.14159265")
        ds.add_new(0x00091102, "US", 2)
        ds.add_new(0x00091103, "US", None)
        ds.add_new(0x00091104, "US", "")
        ds.add_new(0x00091105, "US", [])

        ds_json = ds.to_json_dict()

        assert ds_json["0009100B"]["Value"] == [3000000000]
        assert ds_json["0009100C"]["Value"] == [-2000000000]
        assert ds_json["0009100D"]["Value"] == [40000]
        assert ds_json["0009100E"]["Value"] == [-22222]
        assert ds_json["0009100F"]["Value"] == [3.14]
        assert ds_json["00091010"]["Value"] == [3.14159265]
        assert ds_json["00091014"]["Value"] == [42]
        assert ds_json["00091015"]["Value"] == [3.14159265]
        assert ds_json["00091102"]["Value"] == [2]
        assert "Value" not in ds_json["00091103"]
        assert "Value" not in ds_json["00091104"]
        assert "Value" not in ds_json["00091105"]

    def test_numeric_types(self):
        ds = Dataset()

        ds.add_new(0x0009100B, "UL", 3000000000)
        ds.add_new(0x0009100C, "SL", -2000000000)
        ds.add_new(0x0009100D, "US", 40000)
        ds.add_new(0x0009100E, "SS", -22222)
        ds.add_new(0x0009100F, "FL", 3.14)
        ds.add_new(0x00091010, "FD", 3.14159265)
        ds.add_new(0x00091014, "IS", "42")
        ds.add_new(0x00091015, "DS", "3.14159265")
        ds.add_new(0x00091102, "US", 2)

        ds_json = ds.to_json_dict()

        assert isinstance(ds_json["0009100B"]["Value"][0], int)
        assert isinstance(ds_json["0009100C"]["Value"][0], int)
        assert isinstance(ds_json["0009100D"]["Value"][0], int)
        assert isinstance(ds_json["0009100E"]["Value"][0], int)
        assert isinstance(ds_json["0009100F"]["Value"][0], float)
        assert isinstance(ds_json["00091010"]["Value"][0], float)
        assert isinstance(ds_json["00091014"]["Value"][0], int)
        assert isinstance(ds_json["00091015"]["Value"][0], float)
        assert isinstance(ds_json["00091102"]["Value"][0], int)


class CountingBulkDataLoader:
    """A bulk data URI handler that counts and records its invocations."""

    PIXEL_DATA = bytes(range(256)) * 4096  # 1 MiB of "pixel data"

    def __init__(self):
        self.calls = []

    def __call__(self, tag, vr, uri):
        self.calls.append((tag, vr, uri))
        if uri.endswith("/pixel"):
            return self.PIXEL_DATA

        return f"data-for-{uri.rsplit('/', 1)[-1]}".encode()

    @staticmethod
    def json_dataset():
        """A JSON dataset with a large Pixel Data element and two bulk
        data values nested inside two levels of sequences."""
        return {
            "00100010": {"vr": "PN", "Value": [{"Alphabetic": "Doe^John"}]},
            "7FE00010": {
                "vr": "OW",
                "BulkDataURI": "https://example.com/bulk/pixel",
            },
            "00081115": {
                "vr": "SQ",
                "Value": [
                    {
                        "52009229": {
                            "vr": "SQ",
                            "Value": [
                                {
                                    "00091002": {
                                        "vr": "OB",
                                        "BulkDataURI": "https://example.com/bulk/a",
                                    }
                                },
                                {
                                    "00091002": {
                                        "vr": "OB",
                                        "BulkDataURI": "https://example.com/bulk/b",
                                    }
                                },
                            ],
                        }
                    }
                ],
            },
        }


class TestLazyBulkData:
    """Tests for the lazy resolution of "BulkDataURI" values."""

    def setup_method(self):
        self.loader = CountingBulkDataLoader()
        self.ds = Dataset.from_json(
            json.dumps(CountingBulkDataLoader.json_dataset()),
            bulk_data_uri_handler=self.loader,
        )

    def nested_element(self, index):
        return self.ds[0x00081115].value[0][0x52009229].value[index][0x00091002]

    def test_no_load_on_parse(self):
        # Parsing the JSON must not retrieve any bulk data
        assert self.loader.calls == []
        assert isinstance(self.ds[0x7FE00010]._value, BulkDataValue)
        assert isinstance(self.nested_element(0)._value, BulkDataValue)
        assert isinstance(self.nested_element(1)._value, BulkDataValue)

    def test_lazy_value_remembers_tag_vr_uri_and_path(self):
        lazy = self.ds[0x7FE00010]._value
        assert lazy.tag == "7FE00010"
        assert lazy.vr == "OW"
        assert lazy.uri == "https://example.com/bulk/pixel"
        assert lazy.path == ()
        assert not lazy.is_loaded

        nested = self.nested_element(1)._value
        assert nested.tag == "00091002"
        assert nested.vr == "OB"
        assert nested.uri == "https://example.com/bulk/b"
        assert nested.path == (("00081115", 0), ("52009229", 1))

    def test_no_load_on_introspection(self):
        # repr, tag iteration, copying and len() must not retrieve bulk data
        assert "BulkDataValue" in repr(self.ds)
        assert "BulkDataValue" in str(self.nested_element(0))
        assert len(self.ds) == 3
        assert [elem.tag for elem in self.ds] == [
            Tag(0x00081115),
            Tag(0x00100010),
            Tag(0x7FE00010),
        ]
        assert self.ds[0x7FE00010].VM == 1
        assert not self.ds[0x7FE00010].is_empty
        copy.copy(self.ds)
        copied = copy.deepcopy(self.ds)
        assert self.loader.calls == []

        # A deepcopy shares the lazy value (and its future cache)
        assert copied[0x7FE00010]._value is self.ds[0x7FE00010]._value

    def test_json_roundtrip_keeps_uri_without_download(self):
        # Exporting to JSON must emit the original URI without retrieval
        json_dict = self.ds.to_json_dict()
        assert self.loader.calls == []
        assert json_dict["7FE00010"] == {
            "vr": "OW",
            "BulkDataURI": "https://example.com/bulk/pixel",
        }
        nested = json_dict["00081115"]["Value"][0]["52009229"]["Value"]
        assert nested[0]["00091002"]["BulkDataURI"] == ("https://example.com/bulk/a")
        assert nested[1]["00091002"]["BulkDataURI"] == ("https://example.com/bulk/b")

        # A full round-trip does not trigger any download either
        ds2 = Dataset.from_json(self.ds.to_json(), bulk_data_uri_handler=self.loader)
        assert self.loader.calls == []
        assert isinstance(ds2[0x7FE00010]._value, BulkDataValue)

    def test_load_on_value_access_and_cached(self):
        # The actual trigger points are the element value accesses
        assert self.nested_element(1).value == b"data-for-b"
        assert self.loader.calls == [("00091002", "OB", "https://example.com/bulk/b")]

        # A successfully loaded value is only retrieved once
        assert self.nested_element(1).value == b"data-for-b"
        assert not isinstance(self.nested_element(1)._value, BulkDataValue)
        assert self.nested_element(0).value == b"data-for-a"
        assert self.ds.PixelData == CountingBulkDataLoader.PIXEL_DATA
        assert self.ds.PixelData == CountingBulkDataLoader.PIXEL_DATA
        assert len(self.loader.calls) == 3

    def test_deepcopy_shares_successful_load(self):
        copied = copy.deepcopy(self.ds)
        assert copied[0x7FE00010].value == CountingBulkDataLoader.PIXEL_DATA
        assert len(self.loader.calls) == 1
        # The copy resolved the shared lazy value, the original sees it too
        assert self.ds[0x7FE00010].value == CountingBulkDataLoader.PIXEL_DATA
        assert len(self.loader.calls) == 1

    @pytest.mark.skipif(not have_numpy, reason="NumPy is not available")
    def test_pixel_decode_triggers_load(self):
        import numpy as np

        loader = CountingBulkDataLoader()
        ds = Dataset.from_json(
            {
                "7FE00010": {
                    "vr": "OW",
                    "BulkDataURI": "https://example.com/bulk/pixel",
                }
            },
            bulk_data_uri_handler=loader,
        )
        ds.Rows = 1024
        ds.Columns = 1024
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.file_meta = Dataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        assert loader.calls == []
        arr = ds.pixel_array
        assert len(loader.calls) == 1
        assert arr.shape == (1024, 1024)
        assert arr.dtype == np.uint8
        assert arr[3, 3] == 3

    def test_failed_load_allows_explicit_retry(self):
        attempts = []
        should_fail = [True]

        def flaky_loader(tag, vr, uri):
            attempts.append(uri)
            if should_fail[0]:
                raise RuntimeError("retrieval failed")

            return b"recovered"

        ds = Dataset.from_json(
            {"00091002": {"vr": "OB", "BulkDataURI": "https://example.com/x"}},
            bulk_data_uri_handler=flaky_loader,
        )

        # The failure propagates and the value stays unresolved
        with pytest.raises(RuntimeError, match="retrieval failed"):
            ds[0x00091002].value
        assert len(attempts) == 1
        lazy = ds[0x00091002]._value
        assert isinstance(lazy, BulkDataValue)
        assert not lazy.is_loaded

        # The caller may explicitly retry the failed load
        should_fail[0] = False
        assert lazy.retry() == b"recovered"
        assert lazy.is_loaded
        assert len(attempts) == 2

        # Re-accessing the value after a failure also retries
        should_fail[0] = True
        ds = Dataset.from_json(
            {"00091002": {"vr": "OB", "BulkDataURI": "https://example.com/x"}},
            bulk_data_uri_handler=flaky_loader,
        )
        with pytest.raises(RuntimeError, match="retrieval failed"):
            ds[0x00091002].value
        should_fail[0] = False
        assert ds[0x00091002].value == b"recovered"

    def test_empty_bulk_data_uri(self):
        # An empty URI references no bulk data - no retrieval happens
        ds = Dataset.from_json(
            {"00091002": {"vr": "OB", "BulkDataURI": ""}},
            bulk_data_uri_handler=self.loader,
        )
        assert ds[0x00091002].is_empty
        assert self.loader.calls == []

        # A loader returning empty bytes resolves to an empty value
        ds = Dataset.from_json(
            {"00091002": {"vr": "OB", "BulkDataURI": "https://example.com/e"}},
            bulk_data_uri_handler=lambda tag, vr, uri: b"",
        )
        assert ds[0x00091002].value == b""
        assert ds[0x00091002].is_empty

    def test_unknown_vr(self):
        # An unknown VR gets a lazy value that resolves to bytes and is
        # exported as inline binary once resolved
        ds = Dataset.from_json(
            {"00091002": {"vr": "ZX", "BulkDataURI": "https://example.com/z"}},
            bulk_data_uri_handler=lambda tag, vr, uri: b"zx-bytes",
        )
        elem = ds[0x00091002]
        assert elem.VR == "ZX"
        assert isinstance(elem._value, BulkDataValue)
        assert ds.to_json_dict()["00091002"] == {
            "vr": "ZX",
            "BulkDataURI": "https://example.com/z",
        }

        assert elem.value == b"zx-bytes"
        assert ds.to_json_dict()["00091002"] == {
            "vr": "ZX",
            "InlineBinary": base64.b64encode(b"zx-bytes").decode("utf-8"),
        }

    def test_save_as_writes_real_bytes(self, tmp_path):
        # Saving the dataset must write the retrieved bytes, not the
        # lazy value or its URI
        path = tmp_path / "lazy_bulk.dcm"
        self.ds.save_as(str(path), implicit_vr=False, little_endian=True)
        assert len(self.loader.calls) == 3

        raw = path.read_bytes()
        assert CountingBulkDataLoader.PIXEL_DATA[:64] in raw
        assert b"data-for-a" in raw
        assert b"data-for-b" in raw
        assert b"https://example.com" not in raw

        ds = dcmread(str(path), force=True)
        assert ds.PixelData == CountingBulkDataLoader.PIXEL_DATA
        nested = ds[0x00081115].value[0][0x52009229].value
        assert nested[0][0x00091002].value == b"data-for-a"
        assert nested[1][0x00091002].value == b"data-for-b"
