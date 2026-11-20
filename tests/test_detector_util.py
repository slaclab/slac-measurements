import pytest
from slac_measurements.wires.detector_util import (
    resolve_beampath_key,
    pick_by_beampath,
    resolve_detectors,
)


class TestResolveBeampathKey:
    def test_cu_beampath(self):
        assert resolve_beampath_key("CU_HXR") == "CU"

    def test_sc_beampath(self):
        assert resolve_beampath_key("SC_SXR") == "SC"

    def test_cu_prefix_only(self):
        assert resolve_beampath_key("CU") == "CU"


class TestPickByBeampath:
    def test_dict_cu(self):
        assert pick_by_beampath({"CU": "PMT756", "SC": "LBLM32A"}, "CU_HXR") == "PMT756"

    def test_dict_sc(self):
        assert (
            pick_by_beampath({"CU": "PMT756", "SC": "LBLM32A"}, "SC_HXR") == "LBLM32A"
        )

    def test_dict_missing_key_returns_fallback(self):
        assert pick_by_beampath({"SC": "LBLM32A"}, "CU_HXR", fallback="") == ""

    def test_dict_missing_key_default_fallback_is_none(self):
        assert pick_by_beampath({"SC": "LBLM32A"}, "CU_HXR") is None

    def test_flat_string_passthrough(self):
        assert pick_by_beampath("PMT756", "CU_HXR") == "PMT756"

    def test_flat_list_passthrough(self):
        detectors = ["PMT122", "LBLM32A"]
        assert pick_by_beampath(detectors, "SC_HXR") == detectors


class TestResolveDetectors:
    def test_flat_list_passthrough(self):
        config = resolve_detectors(
            detectors=["PMT122", "LBLM32A"],
            default_detector="PMT122",
            beampath="CU_HXR",
        )
        assert config.names == ["PMT122", "LBLM32A"]
        assert config.default == "PMT122"

    def test_dict_cu_beampath(self):
        config = resolve_detectors(
            detectors={
                "CU": ["PMT122", "LBLM32A"],
                "SC": ["LBLM32A"],
            },
            default_detector={"CU": "PMT122", "SC": "LBLM32A"},
            beampath="CU_HXR",
        )
        assert config.names == ["PMT122", "LBLM32A"]
        assert config.default == "PMT122"

    def test_dict_sc_beampath(self):
        config = resolve_detectors(
            detectors={
                "CU": ["PMT122", "LBLM32A"],
                "SC": ["LBLM32A"],
            },
            default_detector={"CU": "PMT122", "SC": "LBLM32A"},
            beampath="SC_HXR",
        )
        assert config.names == ["LBLM32A"]
        assert config.default == "LBLM32A"

    def test_dict_sc_sxr_beampath(self):
        config = resolve_detectors(
            detectors={
                "CU": ["LBLMS32A"],
                "SC": ["LBLMS32A", "TMITLOSS"],
            },
            default_detector={"CU": "LBLMS32A", "SC": "TMITLOSS"},
            beampath="SC_SXR",
        )
        assert config.names == ["LBLMS32A", "TMITLOSS"]
        assert config.default == "TMITLOSS"

    def test_empty_default_falls_back_to_first_detector(self):
        config = resolve_detectors(
            detectors=["PMT756"],
            default_detector="",
            beampath="CU_HXR",
        )
        assert config.default == "PMT756"

    def test_dict_missing_key_falls_back_to_first_detector(self):
        config = resolve_detectors(
            detectors=["PMT756"],
            default_detector={"SC": "LBLM32A"},
            beampath="CU_HXR",
        )
        assert config.default == "PMT756"

    def test_no_detectors_and_no_default_raises(self):
        with pytest.raises(RuntimeError, match="No detectors available"):
            resolve_detectors(
                detectors={"SC": ["LBLM32A"]},
                default_detector={"SC": "LBLM32A"},
                beampath="CU_HXR",
            )
