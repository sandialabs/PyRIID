# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for working with the GADRAS API."""
import json
import logging
import os
import sys

import tqdm
from jsonschema import validate

from riid.data.sampleset import SampleSet, read_pcf

GADRAS_API_SEEMINGLY_AVAILABLE = False
GADRAS_DIR_ENV_VAR_KEY = "GADRAS_DIR"
GADRAS_INSTALL_PATH = os.getenv(GADRAS_DIR_ENV_VAR_KEY) \
    if GADRAS_DIR_ENV_VAR_KEY in os.environ \
    else "C:\\GADRAS"
GADRAS_ASSEMBLY_PATH = os.path.join(
    GADRAS_INSTALL_PATH,
    "Program"
)
GADRAS_API_CONFIG_FILE_PATH = os.path.join(
    GADRAS_ASSEMBLY_PATH,
    "gadras_api.ini"
)
GADRAS_DETECTOR_DIR_NAME = "Detector"
GADRAS_DETECTOR_DIR_PATH = os.path.join(
    GADRAS_INSTALL_PATH,
    GADRAS_DETECTOR_DIR_NAME
)

GADRAS_API_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "api_schema.json"
)
with open(GADRAS_API_SCHEMA_PATH, "r") as fin:
    GADRAS_API_SCHEMA = json.load(fin)

GADRAS_API_SEEMINGLY_AVAILABLE = os.path.exists(GADRAS_API_CONFIG_FILE_PATH)
IS_GADRAS19 = False

if sys.platform == "win32" and GADRAS_API_SEEMINGLY_AVAILABLE:
    import clr
    sys.path.append(GADRAS_ASSEMBLY_PATH)
    clr.AddReference("Sandia.Gadras.API")
    clr.AddReference("Sandia.Gadras.Utilities")
    clr.AddReference("System.Collections")

    from Sandia.Gadras.API import GadrasAPIWrapper, LocationInfo  # noqa
    try:
        from Sandia.Gadras.API.Inject import InjectSetup  # noqa
        IS_GADRAS19 = True
    except ModuleNotFoundError:
        from Sandia.Gadras.API import InjectSetup  # noqa
    from Sandia.Gadras.Utilities import Configs  # noqa
    from System.Collections.Generic import List  # noqa


INJECT_PARAMS = {
    "DISTANCE_CM": {"type": "float"},
    "HEIGHT_CM": {"type": "float"},
    "DEAD_TIME_PER_PULSE": {"type": "float"},
    "LATITUDE_DEG": {"type": "float"},
    "LONGITUDE_DEG": {"type": "float"},
    "ELEVATION_M": {"type": "float"},
}
DETECTOR_PARAMS = {
    # "DISTANCE_CM": {"type": "float"},  # changed in inject setup instead
    # "HEIGHT_CM": {"type": "float"},  # changed in inject setup instead
    # Detector
    "DET_SETBACK": {"type": "float"},
    "DET_LENGTH_CM": {"type": "float"},
    "DET_WIDTH_CM": {"type": "float"},
    "DET_HEIGHT_WIDTH_RATIO": {"type": "float"},
    "SHAPE_FACTOR": {"type": "float"},
    "DEAD_LAYER": {"type": "float"},
    "EFF_SCALAR": {"type": "float"},
    # Peak Shape
    "ERES_OFFSET": {"type": "float"},
    "ERES_FWHM": {"type": "float"},
    "ERES_POWER": {"type": "float"},
    "ERES_LOW_ESKEW": {"type": "float"},
    "ERES_HIGH_ESKEW": {"type": "float"},
    "LOW_ERES_SKEW_POWER": {"type": "float"},
    "HIGH_ERES_SKEW_POWER": {"type": "float"},
    "LOW_ERES_SKEW_EXTENT": {"type": "float"},
    "HIGH_ERES_SKEW_EXTENT": {"type": "float"},
    # E-cal
    "ECAL_ORD_0": {"type": "float"},
    "ECAL_ORD_1": {"type": "float"},
    "ECAL_ORD_2": {"type": "float"},
    "ECAL_ORD_3": {"type": "float"},
    "ECAL_LOW_E": {"type": "float"},
    # Inner attenuator
    "INNER_AN": {"type": "float"},
    "INNER_AD": {"type": "float"},
    "INNER_POROSITY": {"type": "float"},
    # Outer attenuator
    "OUTER_AN": {"type": "float"},
    "OUTER_AD": {"type": "float"},
    "OUTER_POROSITY": {"type": "float"},
    # Timing
    "SHAPE_TIME": {"type": "float"},
    # "DEAD_TIME_PER_PULSE": {"type": "float"},  # changed in inject setup instead
    # Photon Scatter
    "SCATTER_CLUTTER": {"type": "float"},
    "SCATTER_PROB_AT_0": {"type": "float"},
    "SCATTER_PROB_AT_45": {"type": "float"},
    "SCATTER_PROB_AT_90": {"type": "float"},
    "SCATTER_PROB_AT_135": {"type": "float"},
    "SCATTER_PROB_AT_180": {"type": "float"},
    "SCATTER_FLATTEN_EDGE": {"type": "float"},
    "SCATTER_RATE_TOWARDS_ZERO": {"type": "float"},
    "SCATTER_INCREASE_WITH_E": {"type": "float"},
    "SCATTER_ATTENUATE": {"type": "float"},
    "SCATTER_PREF_ANGLE": {"type": "float"},
    "SCATTER_PREF_ANGLE_DELTA": {"type": "float"},
    "SCATTER_PREF_ANGLE_MAG": {"type": "float"},
    # Air pressure
    "AIR_PRESSURE": {"type": "float"},
    # "ELEVATION": {"type": "float"},
    # Other
    "SOLID_ANGLE_PERCENT": {"type": "float"},
    "DEFAULT_CHANNELS": {"type": "int"},
    # "EFFICIENCY_HOLDER": {"type": "float"},
    # "SHIELD_ANGLE_PERCENT": {"type": "float"},
    # "SHIELD_THICKNESS_CM": {"type": "float"},
    # "DEAD_LAYER_Z": {"type": "float"},
    # "DEAD_LAYER_G_CM2": {"type": "float"},
    # "SHIELD_LLD_KEV": {"type": "float"},
    # "SHIELD_MATERIAL_ID": {"type": "int"},
    # "EXT_ANNIHILATION": {"type": "float"},
    # "XRAY_SOURCE_ID": {"type": "int"},
    # "XRAY_SOURCE2_ID": {"type": "int"},
    # "BETAS": {"type": "float"},
    # "HOLE_MU_TAU": {"type": "float"},
    # "LLD_KEV": {"type": "float"},
    # "LLD_SHARPNESS": {"type": "float"},
    # "FRISCH_GRID": {"type": "float"},
    # "LOCAL_XRAY_MAG": {"type": "float"},
    # "LOCAL_XRAY2_MAG": {"type": "float"},
    # "DETECTOR_TYPE": {"type": "int"},
    # "MIN_ENERGY_KEV": {"type": "int"},
    # "MAX_ENERGY_KEV": {"type": "int"},
    # "INBIN": {"type": "int"},
    # "REBIN": {"type": "int"},
    # "PILEUP": {"type": "int"},
    # "SIDE_SHIELD_PERC_PLUS_X": {"type": "float"},
    # "SIDE_SHIELD_PERC_MINUS_X": {"type": "float"},
    # "SIDE_SHIELD_PERC_PLUS_Y": {"type": "float"},
    # "SIDE_SHIELD_PER_MINUS_Y": {"type": "float"},
    # "SIDE_SHIELD_AN": {"type": "float"},
    # "SIDE_SHIELD_AD": {"type": "float"},
    # "BACK_SHIELD_PERC": {"type": "float"},
    # "BACK_SHIELD_AN": {"type": "float"},
    # "BACK_SHIELD_AD": {"type": "float"},
    # "ALT_COLL_DIAM": {"type": "float"},
    # "AC_SHIELD_COMPTON_CAMERA": {"type": "int"},
    # "NEUTRON_SHIELD_PERC": {"type": "float"},
    # "NEUTRON_REFLECT": {"type": "float"},
    # "NEUTRON_ENVIRONMENT": {"type": "float"},
    # "TEMPLATE_ERROR": {"type": "float"},
    # "SCATTER_ENVIRONMENT": {"type": "float"},
    # "AERIAL_START": {"type": "float"},
    # "AERIAL_STOP": {"type": "float"},
}


class BaseInjector():

    def __init__(self, gadras_api=None) -> None:
        self._check_if_gadras_is_installed()
        self.n_records = 1
        self.gadras_api = gadras_api
        if not gadras_api:
            self.gadras_api = get_gadras_api()

    def _check_if_gadras_is_installed(self):
        if not GADRAS_API_SEEMINGLY_AVAILABLE:
            msg = "GADRAS API not found; no injects can be performed."
            raise GadrasNotInstalledError(msg)


class SourceInjector(BaseInjector):
    """Class to obtain total counts-normalized source gamma spectra using the GADRAS API,
    the purpose of which is to seed other synthesizers.
    """

    def __init__(self, gadras_api=None) -> None:
        super().__init__(gadras_api)

    def _get_inject_setups_for_sources(self, gadras_api, detector, sources, isotope, output_path):
        setups = List[InjectSetup]()
        if not sources:
            return setups
        for i, source in enumerate(sources, start=self.n_records):
            setup = get_inject_setup(
                gadras_api=gadras_api,
                output_path=output_path,
                title=isotope,
                record_num=i,
                source=source,
                detector_distance_to_source_cm=detector["distance_cm"],
                detector_height_cm=detector["height_cm"],
                detector_dead_time_usecs=detector["dead_time_per_pulse"],
                detector_elevation_m=detector["elevation_m"],
                detector_latitude_deg=detector["latitude_deg"],
                detector_longitude_deg=detector["longitude_deg"],
                detector_contains_internal_source=False,  # TODO
            )
            setups.Add(setup)
            self.n_records += 1
        return setups

    def generate(self, config: dict, rel_output_path: str, verbose: bool = False) -> SampleSet:
        """Produce a `SampleSet` containing foreground and/or background seeds using GADRAS based
        on the given inject configuration.

        Args:
            config: dictionary containing the needed information to perform injects
                via the GADRAS API
            verbose: whether to show detailed output

        Returns:
            `SampleSet` containing foreground and/or background seeds generated by GADRAS
        """
        worker = None
        if IS_GADRAS19:
            worker = self.gadras_api.GetBatchInjectWorker()

        injects_exist = False
        source_configs = tqdm.tqdm(config["sources"], desc="Running injects")
        for fg in source_configs:
            inject_setups = self._get_inject_setups_for_sources(
                self.gadras_api,
                config["gamma_detector"]["parameters"],
                fg["configurations"],
                fg["isotope"],
                rel_output_path
            )
            if inject_setups:
                if worker:
                    worker.Run(inject_setups)
                else:
                    for setup in inject_setups:
                        self.gadras_api.injectGenerateData(setup)
                injects_exist = True

        if not injects_exist:
            return

        # Add source name to foreground seeds file
        if verbose:
            logging.info("Filling in source names...")
        record_num = 1
        for fg in config["sources"]:
            for source in fg["configurations"]:
                self.gadras_api.spectraFileWriteSingleField(
                    rel_output_path,
                    record_num,
                    "Source",
                    source
                )
                record_num += 1

        abs_output_path = os.path.join(
            GADRAS_DETECTOR_DIR_PATH,
            config["gamma_detector"]["name"],
            rel_output_path
        )

        if verbose:
            logging.info(f"Sources saved to: {abs_output_path}")

        return abs_output_path


def get_gadras_api(instance_num=1, initialize_transport=True):
    """Initialize the GADRAS API object."""
    if not os.path.exists(GADRAS_API_CONFIG_FILE_PATH):
        Configs.createApiSettingsFile(
            GADRAS_API_CONFIG_FILE_PATH,
            GADRAS_INSTALL_PATH
        )
    api_Settings_loaded = Configs.loadApiSettings(
        GADRAS_API_CONFIG_FILE_PATH,
        createNonExistantDirectories=True
    )
    if not api_Settings_loaded:
        raise RuntimeError("Failed to load API settings.")

    return GadrasAPIWrapper(
        ".",
        instance_num,
        initialize_transport
    )


def get_inject_setup(gadras_api, output_path, title, record_num, source,
                     detector_distance_to_source_cm, detector_height_cm,
                     detector_dead_time_usecs, detector_elevation_m, detector_latitude_deg,
                     detector_longitude_deg, detector_overburden=0, detector_dwell_time_secs=1,
                     detector_dwell_time_is_live_time=True,
                     detector_contains_internal_source=False, include_poisson_variations=False,
                     background_include_cosmic=False, background_include_terrestrial=False,
                     background_K40_percent=0.0, background_U_ppm=0.0,
                     background_Th232_ppm=0.0, background_attenuation=0.0,
                     background_low_energy_continuum=0.0, background_high_energy_continuum=0.0,
                     background_suppression_scalar=0.0):
    """Build a GADRAS `InjectSetup` object."""
    setup = InjectSetup()

    if IS_GADRAS19:
        setup.SetDefaults(gadras_api)
    else:
        setup.setDefaults(gadras_api)

    setup.FileName = output_path
    setup.Title = title
    setup.Record = record_num
    setup.Source = source

    setup.DistanceToSourceCm = detector_distance_to_source_cm
    setup.DetectorHeightCm = detector_height_cm
    setup.DetectorDeadTimeUs = detector_dead_time_usecs
    loc_info = LocationInfo()
    loc_info.Elevation = detector_elevation_m
    loc_info.Latitude = detector_latitude_deg
    loc_info.Longitude = detector_longitude_deg
    loc_info.Overburden = detector_overburden

    if IS_GADRAS19:
        setup.UpdateBackgroundLocation(loc_info)
    else:
        setup.LocationInfo = loc_info

    setup.DwellTimeSec = detector_dwell_time_secs
    setup.DwellTimeIsLiveTime = detector_dwell_time_is_live_time
    setup.ContainsInternalSource = detector_contains_internal_source

    setup.IncludePoissonVariations = include_poisson_variations

    setup.IncludeCosmicBackground = background_include_cosmic
    setup.IncludeTerrestrialBackground = background_include_terrestrial
    if background_include_terrestrial:
        setup.TerrestrialBackground.K40 = background_K40_percent
        setup.TerrestrialBackground.Uranium = background_U_ppm
        setup.TerrestrialBackground.Th232 = background_Th232_ppm
        setup.BackgroundAdjustments.LowEnergyContinuum = background_low_energy_continuum
        setup.BackgroundAdjustments.HighEnergyContinuum = \
            background_high_energy_continuum
        setup.BackgroundAdjustments.Attenuation = background_attenuation
        setup.BackgroundAdjustments.BackgroundSuppressionScalar = \
            background_suppression_scalar

    return setup


def get_source_cps(gadras_api, detector, worker, source):
    """Obtain the count rate for a detector and source inject string."""
    rel_output_path = f"{source}.pcf"
    inject_setup = get_inject_setup(
        gadras_api=gadras_api,
        output_path=rel_output_path,
        title=source,
        record_num=1,
        source=None,
        detector_dwell_time_secs=1,
        detector_dwell_time_is_live_time=True,
        detector_distance_to_source_cm=detector["distance_cm"],
        detector_height_cm=detector["height_cm"],
        detector_dead_time_usecs=detector["dead_time_per_pulse"],
        detector_elevation_m=detector["elevation_m"],
        detector_latitude_deg=detector["latitude_deg"],
        detector_longitude_deg=detector["longitude_deg"]
    )
    setups = List[InjectSetup]()
    setups.Add(inject_setup)
    worker.Run(setups)
    ss = read_pcf(rel_output_path)
    os.remove(rel_output_path)
    # It's technically a count rate because the live time was 1
    source_cps = ss.spectra.iloc[0].sum()
    return source_cps


def validate_inject_config(config: dict):
    validate(instance=config, schema=GADRAS_API_SCHEMA)


class GadrasNotInstalledError(Exception):
    """PyRIID could not find a GADRAS installation."""
    pass
