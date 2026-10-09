import h5py
import slac_tools.pydantic_h5
from pydantic import BaseModel, ConfigDict

from slac_measurements.beam_profile import (
    BeamProfileMeasurementResult,
)
from slac_measurements.utils import NDArrayAnnotatedType
from slac_measurements.wires.collection.results import (
    WireMeasurementCollectionResult,
    _read_scan_ranges,
    _write_scan_ranges,
)


class DetectorProfileMeasurement(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    values: NDArrayAnnotatedType
    units: str | None = None
    label: str | None = None


class ProfileMeasurement(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    positions: NDArrayAnnotatedType
    detectors: dict[str, DetectorProfileMeasurement]
    profile_indices: NDArrayAnnotatedType


class DetectorFit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mean: float
    sigma: float
    amplitude: float
    offset: float
    curve: NDArrayAnnotatedType
    positions: NDArrayAnnotatedType


class FitResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    detectors: dict[str, DetectorFit]


class WireMeasurementAnalysisResult(BeamProfileMeasurementResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fit_result: dict[str, FitResult]
    collection_result: WireMeasurementCollectionResult
    profiles: dict[str, ProfileMeasurement]
    fitting_method: str = "gaussian"
    jitter_corrected: bool = False
    jitter_rms: tuple[float, float] | None = None
    charge_normalized: bool = False
    charge_toroid: str | None = None

    def to_mat(self, filepath: str, **kwargs) -> str:
        """Export this result as a MATLAB .mat file compatible with wirescan_gui."""
        from slac_measurements.wires.analysis.mat_export import analysis_result_to_mat

        return analysis_result_to_mat(self, filepath, **kwargs)

    def reanalyze(
        self,
        jitter_correction: bool = False,
        charge_normalization: bool = False,
        charge_toroid: str | None = None,
        fitting_method: str = "gaussian",
        rms_detector: str | None = None,
        physics_model: str = "BLEM",
    ) -> "WireMeasurementAnalysisResult":
        """Re-analyze the collected data with different settings.

        Parameters
        ----------
        jitter_correction : bool
            If True, apply orbit-fit jitter correction.
        charge_normalization : bool
            If True, normalize detector signals by per-pulse charge.
        charge_toroid : str, optional
            Toroid device name for charge normalization.
        fitting_method : str
            Fitting method to use. Default "gaussian".
        rms_detector : str, optional
            Override detector for RMS sizes.
        physics_model : str
            Model source for R-matrix retrieval. Default "BLEM".

        Returns
        -------
        WireMeasurementAnalysisResult
            New analysis result from the same raw collection data.
        """
        from slac_measurements.wires.analysis import WireMeasurementAnalysis

        analysis = WireMeasurementAnalysis(
            collection_result=self.collection_result,
            fitting_method=fitting_method,
        )
        return analysis.analyze(
            rms_detector=rms_detector,
            jitter_correction=jitter_correction,
            charge_normalization=charge_normalization,
            charge_toroid=charge_toroid,
            physics_model=physics_model,
        )

    def __repr__(self) -> str:
        """Return a compact string representation of the analysis result."""
        meta = self.collection_result.metadata
        profile_count = len(self.profiles)
        fit_profile_count = len(self.fit_result)
        detector_count = len(meta.detectors)
        rms_sizes_repr = self.rms_sizes.tolist() if self.rms_sizes is not None else None

        return (
            f"WireMeasurementAnalysisResult("
            f"wire_name='{meta.wire_name}', "
            f"beampath='{meta.beampath}', "
            f"rms_sizes={rms_sizes_repr}, "
            f"rms_detector='{meta.rms_detector}', "
            f"fitting_method='{self.fitting_method}', "
            f"jitter_corrected={self.jitter_corrected}, "
            f"jitter_rms={self.jitter_rms}, "
            f"charge_normalized={self.charge_normalized}, "
            f"charge_toroid={self.charge_toroid!r}, "
            f"profiles={profile_count}, "
            f"fit_profiles={fit_profile_count}, "
            f"detectors={detector_count}, "
            f"timestamp={meta.timestamp.isoformat() if meta.timestamp is not None else None})"
        )

    def save_to_h5(self, filepath: str) -> None:
        """
        Persist the analysis result to an HDF5 file.

        The structure is designed to bundle both the raw collection
        information and the derived analysis data so that a single file
        contains everything necessary to reproduce or inspect a
        wire-scan analysis.  The layout mirrors the one used by
        :py:meth:`WireMeasurementCollectionResult.save_to_h5` for the
        collection portion and adds two additional groups under
        ``/analysis``:
            Hierarchical groups for each profile and detector containing
            fit parameters, the fitted curve and the positions array.
        ``/analysis/profiles``
            Raw profile measurements (positions, detector values and
            indices) used as input to the fitting routine.

        Parameters
        ----------
        filepath : str
            Path where the HDF5 file will be written.  Any existing file
            at that location will be overwritten.
        """

        def _write_fit_result(value: dict, group: h5py.Group) -> None:
            # Backward compatibility for existing hdf5 files:
            # FitResult is a one-field {"detectors": {...}} wrapper; flatten
            # past it so detectors sit directly under each profile, matching
            # the original layout
            fit_grp = group.create_group("analysis/fit_result")
            for profile, fit in value.items():
                prof_grp = fit_grp.create_group(profile)
                for det_name, det_fit in fit.detectors.items():
                    slac_tools.pydantic_h5.save_model(
                        det_fit, prof_grp.create_group(det_name)
                    )

        with h5py.File(filepath, "w") as f:
            slac_tools.pydantic_h5.save_model(
                self,
                f,
                name_map={"profiles": "analysis/profiles"},
                manual={"scan_ranges": _write_scan_ranges, "fit_result": _write_fit_result},
                skip={"metadata"},  # duplicate of collection_result.metadata
            )


def load_from_h5(filepath: str) -> WireMeasurementAnalysisResult:
    """
    Load a :class:`WireMeasurementAnalysisResult` previously written with
    :meth:`WireMeasurementAnalysisResult.save_to_h5`.

    This helper mirrors the on-disk structure defined above.  It is
    primarily intended for unit tests but may also be useful for
    debugging and post-processing scripts.
    """

    def _read_fit_result(group: h5py.Group) -> dict[str, FitResult] | None:
        # Backward compatibility for existing hdf5 files:
        # Inverse of `_write_fit_result`: re-insert the FitResult wrapper
        # the writer flattened away
        if "analysis/fit_result" not in group:
            return None
        fit_grp = group["analysis/fit_result"]
        return {
            profile: FitResult(
                detectors={
                    det: slac_tools.pydantic_h5.load_model(DetectorFit, prof_grp[det])
                    for det in prof_grp.keys()
                }
            )
            for profile, prof_grp in ((p, fit_grp[p]) for p in fit_grp.keys())
        }

    with h5py.File(filepath, "r") as f:
        return slac_tools.pydantic_h5.load_model(
            WireMeasurementAnalysisResult,
            f,
            name_map={"profiles": "analysis/profiles"},
            manual={"scan_ranges": _read_scan_ranges, "fit_result": _read_fit_result},
            extra={"metadata": lambda resolved: resolved["collection_result"].metadata},
        )
