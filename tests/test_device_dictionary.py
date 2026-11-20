from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from slac_devices.wire import Wire
from slac_measurements.wires.collection.step import StepWireMeasurementCollection


def _make_collection(detectors, jitter_bpms=None, charge_toroids=None):
    metadata = MagicMock()
    metadata.jitter_bpms = jitter_bpms
    metadata.charge_toroids = charge_toroids
    wire = Wire.model_construct(name="WSDG01", metadata=metadata)
    return StepWireMeasurementCollection.model_construct(
        beam_profile_device=wire,
        beampath="SC_DIAG0",
        detectors=detectors,
        logger=MagicMock(),
        buffer=MagicMock(),
    )


_DB = {
    "LBLM01A": SimpleNamespace(area="HTR"),
    "PMT756": SimpleNamespace(area="LTUH"),
    "PMT755": SimpleNamespace(area="LTUH"),
    "BPMDG001": SimpleNamespace(area="DIAG0"),
    "BPMDG000": SimpleNamespace(area="COL0"),
    "IM01": SimpleNamespace(area="DIAG0"),
}


def _fake_create_device(name):
    # PMT755 mimics a device that fails validation (PVs missing in slac_db).
    return None if name == "PMT755" else f"device:{name}"


@patch(
    "slac_db.device.build_many_devices",
    side_effect=lambda names: {n: _DB[n] for n in names if n in _DB},
)
@patch("slac_devices.reader.create_device", side_effect=_fake_create_device)
class CreateDeviceDictionaryTest(TestCase):
    def test_devices_created_by_name(self, mock_create, _db):
        c = _make_collection(
            ["LBLM01A", "PMT756"],
            jitter_bpms=["BPMDG001", "BPMDG000"],
            charge_toroids=["IM01"],
        )
        devices = c._create_device_dictionary()
        names = ["LBLM01A", "PMT756", "BPMDG001", "BPMDG000", "IM01"]
        self.assertEqual(list(devices), ["WSDG01", *names])
        self.assertEqual(
            [call.args for call in mock_create.call_args_list],
            [(n,) for n in names],
        )
        self.assertEqual(devices["BPMDG000"], "device:BPMDG000")

    def test_single_db_lookup_with_duplicates_removed(self, mock_create, mock_db):
        c = _make_collection(
            ["BPMDG001"], jitter_bpms=["BPMDG001"], charge_toroids=["BPMDG001"]
        )
        devices = c._create_device_dictionary()
        mock_db.assert_called_once_with(["BPMDG001"])
        mock_create.assert_called_once_with("BPMDG001")
        self.assertEqual(list(devices), ["WSDG01", "BPMDG001"])

    def test_missing_devices_skipped_without_create(self, mock_create, _db):
        c = _make_collection(["SBLM01A", "LBLM01A"], jitter_bpms=["PMT850"])
        devices = c._create_device_dictionary()
        self.assertEqual(list(devices), ["WSDG01", "LBLM01A"])
        mock_create.assert_called_once_with("LBLM01A")
        warned = [call.args[1] for call in c.logger.warning.call_args_list]
        self.assertEqual(warned, ["SBLM01A", "PMT850"])

    def test_failed_creation_is_skipped(self, _create, _db):
        c = _make_collection(["PMT755", "PMT756"])
        devices = c._create_device_dictionary()
        self.assertEqual(list(devices), ["WSDG01", "PMT756"])
        c.logger.warning.assert_called_once()

    @patch("slac_measurements.tmit_loss.TMITLoss", return_value="tmitloss")
    def test_tmitloss_is_built_without_db_lookup(self, _tmit, mock_create, mock_db):
        c = _make_collection(["TMITLOSS", "LBLM01A"])
        devices = c._create_device_dictionary()
        mock_db.assert_called_once_with(["LBLM01A"])
        mock_create.assert_called_once_with("LBLM01A")
        self.assertEqual(devices["TMITLOSS"], "tmitloss")
