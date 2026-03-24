from typing import Dict

from pydantic import BaseModel, ConfigDict

from slac_measurements.beam_profile import (
    BeamProfileMeasurementResult,
)
from slac_measurements.utils import NDArrayAnnotatedType
from slac_measurements.wires.ws_collection_results import (
    WireMeasurementCollectionResult,
    # helper functions used when loading analysis results
    _load_metadata,
    _load_raw_data,
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
    detectors: Dict[str, DetectorFit]


class WireMeasurementAnalysisResult(BeamProfileMeasurementResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fit_result: Dict[str, FitResult]
    collection_result: WireMeasurementCollectionResult
    profiles: Dict[str, ProfileMeasurement]

    def __repr__(self) -> str:
        """Return a compact string representation of the analysis result."""
        meta = self.collection_result.metadata
        profile_count = len(self.profiles)
        fit_profile_count = len(self.fit_result)
        detector_count = len(meta.detectors)
        rms_sizes_repr = (
            self.rms_sizes.tolist() if self.rms_sizes is not None else None
        )

        return (
            f"WireMeasurementAnalysisResult("
            f"wire_name='{meta.wire_name}', "
            f"beampath='{meta.beampath}', "
            f"rms_sizes={rms_sizes_repr}, "
            f"profiles={profile_count}, "
            f"fit_profiles={fit_profile_count}, "
            f"detectors={detector_count}, "
            f"timestamp={meta.timestamp.isoformat()})"
        )

    def save_to_h5(self, filepath: str) -> None:
        """
        Persist the analysis result to an HDF5 file.

        The structure is designed to bundle both the raw collection
        information and the derived analysis data so that a single file
        contains everything necessary to reproduce or inspect a
        wire‑scan analysis.  The layout mirrors the one used by
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

        # reuse imports locally to avoid adding h5py/np at top-level when
        # the module is imported purely for type information
        import h5py
        import numpy as np

        with h5py.File(filepath, "w") as f:
            # store the collection result under its own subgroup so that
            # standalone collection loaders can still operate if needed
            col_grp = f.create_group("collection_result")

            # metadata and raw_data helpers are defined on the
            # WireMeasurementCollectionResult class; we can just call
            # them directly since we know the instance type.
            meta_grp = col_grp.create_group("metadata")
            self.collection_result._save_metadata(meta_grp)

            # include raw detector data from the collection
            raw_grp = col_grp.create_group("raw_data")
            self.collection_result._save_raw_data(raw_grp)

            # (above block already handled the inherited profile fields)
            # save the beam profile measurement fields inherited from
            # BeamProfileMeasurementResult
            if self.rms_sizes is not None:
                f.create_dataset("rms_sizes", data=np.array(self.rms_sizes))
            if self.centroids is not None:
                f.create_dataset("centroids", data=np.array(self.centroids))
            if self.total_intensities is not None:
                f.create_dataset(
                    "total_intensities", data=np.array(self.total_intensities)
                )
            if self.signal_to_noise_ratios is not None:
                f.create_dataset(
                    "signal_to_noise_ratios",
                    data=np.array(self.signal_to_noise_ratios),
                )

            # analysis-specific data
            analysis_grp = f.create_group("analysis")

            # fit results
            fit_grp = analysis_grp.create_group("fit_result")
            for profile, fit in self.fit_result.items():
                prof_grp = fit_grp.create_group(profile)
                for det_name, det_fit in fit.detectors.items():
                    det_grp = prof_grp.create_group(det_name)
                    det_grp.attrs["mean"] = det_fit.mean
                    det_grp.attrs["sigma"] = det_fit.sigma
                    det_grp.attrs["amplitude"] = det_fit.amplitude
                    det_grp.attrs["offset"] = det_fit.offset
                    det_grp.create_dataset("curve", data=det_fit.curve)
                    det_grp.create_dataset("positions", data=det_fit.positions)

            # profiles
            profs_grp = analysis_grp.create_group("profiles")
            for profile, prof in self.profiles.items():
                pgrp = profs_grp.create_group(profile)
                pgrp.create_dataset("positions", data=prof.positions)
                pgrp.create_dataset(
                    "profile_indices", data=prof.profile_indices
                )
                dets_grp = pgrp.create_group("detectors")
                for det_name, det in prof.detectors.items():
                    dg = dets_grp.create_group(det_name)
                    dg.create_dataset("values", data=det.values)
                    if det.units is not None:
                        dg.attrs["units"] = det.units
                    if det.label is not None:
                        dg.attrs["label"] = det.label


def load_from_h5(filepath: str) -> WireMeasurementAnalysisResult:
    """
    Load a :class:`WireMeasurementAnalysisResult` previously written with
    :meth:`WireMeasurementAnalysisResult.save_to_h5`.

    This helper mirrors the on‑disk structure defined above.  It is
    primarily intended for unit tests but may also be useful for
    debugging and post‑processing scripts.
    """

    import h5py
    import numpy as np

    # first load the collection result using the existing loader by
    # temporarily writing the subgroup to a temporary file-like object.
    with h5py.File(filepath, "r") as f:
        # reuse existing functions
        col_group = f["collection_result"]
        metadata = _load_metadata(col_group["metadata"])
        raw_data = _load_raw_data(col_group["raw_data"])

        # beam profile fields
        rms = f.get("rms_sizes")
        rms_val = tuple(rms[:]) if rms is not None else None
        centroids = f.get("centroids")
        cent_val = centroids[:] if centroids is not None else None
        totint = f.get("total_intensities")
        tot_val = totint[:] if totint is not None else None
        snr = f.get("signal_to_noise_ratios")
        snr_val = snr[:] if snr is not None else None

        # analysis
        analysis_grp = f["analysis"]

        fit_result: Dict[str, FitResult] = {}
        for profile in analysis_grp["fit_result"].keys():
            prof_grp = analysis_grp["fit_result"][profile]
            dets: Dict[str, DetectorFit] = {}
            for det_name in prof_grp.keys():
                dg = prof_grp[det_name]
                dets[det_name] = DetectorFit(
                    mean=float(dg.attrs["mean"]),
                    sigma=float(dg.attrs["sigma"]),
                    amplitude=float(dg.attrs["amplitude"]),
                    offset=float(dg.attrs["offset"]),
                    curve=np.array(dg["curve"]),
                    positions=np.array(dg["positions"]),
                )
            fit_result[profile] = FitResult(detectors=dets)

        profiles: Dict[str, ProfileMeasurement] = {}
        for profile in analysis_grp["profiles"].keys():
            pgrp = analysis_grp["profiles"][profile]
            detectors = {}
            for det_name in pgrp["detectors"].keys():
                dg = pgrp["detectors"][det_name]
                detectors[det_name] = DetectorProfileMeasurement(
                    values=np.array(dg["values"]),
                    units=dg.attrs.get("units", None),
                    label=dg.attrs.get("label", None),
                )
            profiles[profile] = ProfileMeasurement(
                positions=np.array(pgrp["positions"]),
                profile_indices=np.array(pgrp["profile_indices"]),
                detectors=detectors,
            )

    return WireMeasurementAnalysisResult(
        fit_result=fit_result,
        collection_result=WireMeasurementCollectionResult(
            raw_data=raw_data, metadata=metadata
        ),
        profiles=profiles,
        rms_sizes=rms_val,
        centroids=cent_val,
        total_intensities=tot_val,
        signal_to_noise_ratios=snr_val,
        metadata=metadata,
    )
