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
    "BPMDG001": SimpleNamespace(area="DIAG0"),
    "BPMDG000": SimpleNamespace(area="COL0"),
    "IM01": SimpleNamespace(area="DIAG0"),
    "RFBDG001": SimpleNamespace(area="*DIAG0"),
}


@patch(
    "slac_db.device.build_many_devices",
    side_effect=lambda names: {n: _DB[n] for n in names if n in _DB},
)
@patch(
    "slac_devices.reader.create_toroid",
    side_effect=lambda area, name: ("toroid", area, name),
)
@patch(
    "slac_devices.reader.create_bpm", side_effect=lambda area, name: ("bpm", area, name)
)
@patch(
    "slac_devices.reader.create_pmt", side_effect=lambda area, name: ("pmt", area, name)
)
@patch(
    "slac_devices.reader.create_lblm",
    side_effect=lambda area, name: ("lblm", area, name),
)
class CreateDeviceDictionaryTest(TestCase):
    def test_areas_come_from_device_db(self, *_):
        c = _make_collection(
            ["LBLM01A", "PMT756"],
            jitter_bpms=["BPMDG001", "BPMDG000"],
            charge_toroids=["IM01"],
        )
        devices = c._create_device_dictionary()
        self.assertEqual(
            list(devices),
            ["WSDG01", "LBLM01A", "PMT756", "BPMDG001", "BPMDG000", "IM01"],
        )
        self.assertEqual(devices["LBLM01A"], ("lblm", "HTR", "LBLM01A"))
        self.assertEqual(devices["PMT756"], ("pmt", "LTUH", "PMT756"))
        # Jitter BPMs and toroids use their own DB area, not the wire's.
        self.assertEqual(devices["BPMDG000"], ("bpm", "COL0", "BPMDG000"))
        self.assertEqual(devices["IM01"], ("toroid", "DIAG0", "IM01"))

    def test_single_db_lookup_with_duplicates_removed(self, _l, _p, _b, _t, mock_db):
        c = _make_collection(
            ["BPMDG001"], jitter_bpms=["BPMDG001"], charge_toroids=["BPMDG001"]
        )
        devices = c._create_device_dictionary()
        mock_db.assert_called_once_with(["BPMDG001"])
        self.assertEqual(list(devices), ["WSDG01", "BPMDG001"])

    def test_missing_and_starred_devices_are_skipped(self, *_):
        c = _make_collection(["SBLM01A", "LBLM01A"], jitter_bpms=["RFBDG001"])
        devices = c._create_device_dictionary()
        self.assertEqual(list(devices), ["WSDG01", "LBLM01A"])
        warnings = [call.args[1] for call in c.logger.warning.call_args_list]
        self.assertEqual(warnings, ["SBLM01A", "RFBDG001"])

    @patch("slac_measurements.tmit_loss.TMITLoss", return_value="tmitloss")
    def test_tmitloss_is_built_without_db_lookup(self, _tmit, _l, _p, _b, _t, mock_db):
        c = _make_collection(["TMITLOSS", "LBLM01A"])
        devices = c._create_device_dictionary()
        mock_db.assert_called_once_with(["LBLM01A"])
        self.assertEqual(devices["TMITLOSS"], "tmitloss")
