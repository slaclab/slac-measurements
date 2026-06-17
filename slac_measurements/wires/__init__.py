# Primary entry point — most callers only need this
from slac_measurements.wires.scan import WireBeamProfileMeasurement

# Analysis — for re-analyzing a saved collection result without re-running hardware
from slac_measurements.wires.analysis import FittingMethod, WireMeasurementAnalysis

# Result types — for type hints, isinstance checks, and loading saved files
from slac_measurements.wires.analysis_results import (
    DetectorFit,
    DetectorProfileMeasurement,
    FitResult,
    ProfileMeasurement,
    WireMeasurementAnalysisResult,
    load_from_h5 as load_analysis_from_h5,
)
from slac_measurements.wires.collection_results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
    load_from_h5 as load_collection_from_h5,
)

# Collection factory and mode type — for advanced users constructing collections directly
from slac_measurements.wires.collection import ScanMode, create_wire_collection

# Jitter correction — compute beam jitter from BPM data
from slac_measurements.wires.jitter_correction import compute_jitter
