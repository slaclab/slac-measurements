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


# Devices create_many_devices can build; anything else is unknown or invalid.
_CREATABLE = {"LBLM01A", "PMT756", "BPMDG001", "BPMDG000", "IM01"}


def _fake_create_many_devices(names):
    return {n: f"device:{n}" for n in names if n in _CREATABLE}


@patch(
    "slac_devices.reader.create_many_devices",
    side_effect=_fake_create_many_devices,
)
class CreateDeviceDictionaryTest(TestCase):
    def test_devices_created_in_one_call(self, mock_create):
        c = _make_collection(
            ["LBLM01A", "PMT756"],
            jitter_bpms=["BPMDG001", "BPMDG000"],
            charge_toroids=["IM01"],
        )
        devices = c._create_device_dictionary()
        names = ["LBLM01A", "PMT756", "BPMDG001", "BPMDG000", "IM01"]
        mock_create.assert_called_once_with(names)
        self.assertEqual(list(devices), ["WSDG01", *names])
        self.assertEqual(devices["BPMDG000"], "device:BPMDG000")

    def test_duplicates_removed(self, mock_create):
        c = _make_collection(
            ["BPMDG001"], jitter_bpms=["BPMDG001"], charge_toroids=["BPMDG001"]
        )
        devices = c._create_device_dictionary()
        mock_create.assert_called_once_with(["BPMDG001"])
        self.assertEqual(list(devices), ["WSDG01", "BPMDG001"])

    def test_uncreated_devices_skipped_with_warning(self, _create):
        c = _make_collection(["SBLM01A", "LBLM01A"], jitter_bpms=["PMT755"])
        devices = c._create_device_dictionary()
        self.assertEqual(list(devices), ["WSDG01", "LBLM01A"])
        warned = [call.args[1] for call in c.logger.warning.call_args_list]
        self.assertEqual(warned, ["SBLM01A", "PMT755"])

    @patch("slac_measurements.tmit_loss.TMITLoss", return_value="tmitloss")
    def test_tmitloss_built_without_lookup(self, _tmit, mock_create):
        c = _make_collection(["TMITLOSS", "LBLM01A"])
        devices = c._create_device_dictionary()
        mock_create.assert_called_once_with(["LBLM01A"])
        self.assertEqual(list(devices), ["WSDG01", "TMITLOSS", "LBLM01A"])
        self.assertEqual(devices["TMITLOSS"], "tmitloss")
