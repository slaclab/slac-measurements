from typing import Dict, Union

import numpy as np

from slac_devices.magnet import Magnet

from slac_devices.screen import Screen
from slac_devices.wire import Wire
from slac_measurements.beam_profile import BeamProfileMeasurement


def quad_scan_optics(
    magnet: Magnet, beam_profile_device: Union[Screen, Wire], physics_model="BMAD"
) -> Dict:
    """Get rmat (6 x 6) from magnet to measurement device and twiss at measurement device"""
    # TODO: get optics from arbitrary devices (potentially in different beam lines)
    # have live BLEM model update
    if physics_model == "BLEM":
        refresh_blem_model()
    model_live = _get_model_from_device(beam_profile_device, physics_model, use_design=False)
    rmat = model_live.get_rmat(
        from_device=magnet.name,
        to_device=beam_profile_device.name,
    )
    model_design = _get_model_from_device(beam_profile_device, physics_model, use_design=True)
    twiss = model_design.get_twiss(beam_profile_device.name)
    return {"rmat": rmat, "design_twiss": twiss}


def get_optics_after_magnet(
    magnet: Magnet, beam_profile_device: Union[Screen, Wire], physics_model="BMAD"
) -> Dict:
    """Get rmat from end of magnet to measurement device"""
    # have live BLEM model update
    if physics_model == "BLEM":
        refresh_blem_model()
    model_live = _get_model_from_device(beam_profile_device, physics_model, use_design=False)
    full_rmat = model_live.get_rmat(
        from_device=magnet.name,
        to_device=beam_profile_device.name,
    )
    quad_rmat = model_live.get_rmat(
        from_device=magnet.name,
        to_device=magnet.name,
    )
    after_quad_rmat = full_rmat @ np.linalg.inv(quad_rmat)
    model_design = _get_model_from_device(beam_profile_device, physics_model, use_design=True)
    twiss = model_design.get_twiss(beam_profile_device.name)
    return {"after_quad_rmat": after_quad_rmat, "design_twiss": twiss}


def multi_device_optics(
    beam_profile_devices: list[Union[Screen, Wire]], physics_model="BMAD"
) -> Dict:
    """Get rmat and twiss from reference device to all measurement devices"""
    if physics_model == "BLEM":
        refresh_blem_model()
    model_live = _get_model_from_device(beam_profile_devices[-1], physics_model, use_design=False)
    beam_profile_device_names = [
        beam_profile_device.name for beam_profile_device in beam_profile_devices
    ]
    rmat = []
    ref_index = int(len(beam_profile_device_names) / 2)
    device_ref = beam_profile_device_names[ref_index]
    for device in beam_profile_device_names:
        rmat.append(model_live.get_rmat(device_ref, device))
    rmat = np.array(rmat)
    model_design = _get_model_from_device(beam_profile_devices[-1], physics_model, use_design=True)
    twiss = model_design.get_twiss(beam_profile_device_names)
    return {"rmat": rmat, "design_twiss": twiss}


def refresh_blem_model():
    from epics import PV
    import threading

    done = threading.Event()

    def on_change(pvname=None, value=None, **kwargs):
        if value == 0:
            done.set()

    # writing 1 to model ctrl PV causes BLEM model to update
    model_ctrl_pv = PV("BLEM:SYS0:1:MAT_MODEL:CTRL")
    model_ctrl_pv.add_callback(on_change)
    model_ctrl_pv.put(1, wait=True)  # blocks until write has processed
    done.wait()  # blocks until ctrl PV has reset to 0


def _get_model_from_device(device, physics_model, use_design=False):
    from meme.model import Model

    # Look for device beam path in meme beam paths
    beam_path = None
    for bp in device.metadata.beam_path:
        if bp in [
            "CU_HXR",
            "CU_SXR",
            "CU_SPEC",
            "SC_DIAG0",
            "SC_BSYD",
            "SC_HXR",
            "SC_SXR",
            "FACET2E",
        ]:
            beam_path = bp
            break
    if beam_path is None:
        raise ValueError("Valid beam path not found in device metadata.")

    return Model(beam_path, model_source=physics_model, use_design=use_design)
