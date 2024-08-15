# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for working with the GADRAS API."""
import copy
import itertools
import json
import os
import sys
from typing import List

import numpy as np
import tqdm
from jsonschema import validate
from numpy.random import Generator

from riid import SampleSet, read_pcf
from riid.data.synthetic.base import get_distribution_values

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
    "PILEUP": {"type": "int"},
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
        pbar = tqdm.tqdm(config["sources"], desc="Running injects")
        for fg in pbar:
            pbar.set_description(f"Running inject for '{fg['isotope']}'")
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
            print("Filling in source names...")
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
            print(f"Sources saved to '{abs_output_path}'")

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
    return validate(instance=config, schema=GADRAS_API_SCHEMA)


def _get_samples_from_dict(
    parameters: dict, rng: Generator = np.random.default_rng()
) -> list:
    keys = list(parameters.keys())
    if keys == ["mean", "std", "num_samples"]:
        return rng.normal(
            parameters["mean"], parameters["std"], parameters["num_samples"]
        ).tolist()
    elif keys == ["min", "max", "dist", "num_samples"]:
        if parameters["dist"] == "uniform":
            return get_distribution_values(
                "uniform",
                (parameters["min"], parameters["max"]),
                n_values=parameters["num_samples"],
                rng=rng,
            ).tolist()
        elif parameters["dist"] == "log10":
            return get_distribution_values(
                "log10",
                (parameters["min"], parameters["max"]),
                n_values=parameters["num_samples"],
                rng=rng,
            ).tolist()


def _compile_single_source_config(
    name: str = "",
    activity=None,
    activity_units: str = "",
    shielding_atomic_number=None,
    shielding_aerial_density=None,
    rng: Generator = np.random.default_rng(),
):
    if isinstance(activity, dict):
        activity = _get_samples_from_dict(activity)
    if isinstance(shielding_atomic_number, dict):
        shielding_atomic_number = _get_samples_from_dict(shielding_atomic_number)
    if isinstance(shielding_aerial_density, dict):
        shielding_aerial_density = _get_samples_from_dict(shielding_aerial_density)

    config_string = f"{name},{float(activity)}{activity_units}"
    if shielding_aerial_density is not None and shielding_atomic_number is not None:
        config_string += f"{{ad={float(shielding_aerial_density)},"
        config_string += f"an={float(shielding_atomic_number)}}}"
    else:
        if shielding_aerial_density is not None:
            config_string += f"{{ad={float(shielding_aerial_density)}}}"
        if shielding_atomic_number is not None:
            config_string += f"{{an={float(shielding_atomic_number)}}}"

    # TODO: check if config string is valid
    return config_string


def _compile_source_configs(
    name: str = "",
    activity=None,
    activity_units: str = "",
    shielding_atomic_number=None,
    shielding_aerial_density=None,
    rng: Generator = np.random.default_rng(),
):
    # check to see if we need to have multiple configurations
    if type(activity) is dict:
        activity = _get_samples_from_dict(activity, rng)
    if type(activity) is not list:
        activity = [activity]
    if type(shielding_atomic_number) is dict:
        shielding_atomic_number = _get_samples_from_dict(shielding_atomic_number, rng)
    if type(shielding_atomic_number) is not list:
        shielding_atomic_number = [shielding_atomic_number]
    if type(shielding_aerial_density) is dict:
        shielding_aerial_density = _get_samples_from_dict(shielding_aerial_density, rng)
    if type(shielding_aerial_density) is not list:
        shielding_aerial_density = [shielding_aerial_density]

    configs = []
    for single_activity in activity:
        for single_an in shielding_atomic_number:
            for single_ad in shielding_aerial_density:
                config_string = _compile_single_source_config(
                    name,
                    single_activity,
                    activity_units,
                    single_an,
                    single_ad,
                    rng,
                )
                configs.append(config_string)

    return configs


def get_expanded_config(config: dict) -> dict:
    """Expands sampling objects within the given config into value lists
    and returns a new, equivalent config dictionary.
    """
    expanded_config = copy.deepcopy(config)

    random_seed = config["random_seed"] if "random_seed" in config else None
    rng = np.random.default_rng(random_seed)

    # detector configs
    for parameter in config["gamma_detector"]["parameters"]:
        parameter_list = []
        parameter_type = type(config["gamma_detector"]["parameters"][parameter])
        parameter_value = config["gamma_detector"]["parameters"][parameter]
        if parameter_type != list:
            if parameter_type == int or parameter_type == float:
                parameter_list.append(parameter_value)
            if parameter_type == dict:
                parameter_list.extend(_get_samples_from_dict(parameter_value, rng))
        else:  # its a list, pull out all items and convert to one list
            for item in parameter_value:
                item_type = type(item)
                if item_type == int or item_type == float:
                    parameter_list.append(item)
                if item_type == dict:
                    parameter_list.extend(_get_samples_from_dict(item, rng))

        # Save distance_cm as a list
        expanded_config["gamma_detector"]["parameters"][parameter] = parameter_list

    # sources configs
    for i, isotope in enumerate(config["sources"]):
        config_list = []
        for config in isotope["configurations"]:
            config_type = type(config)
            if config_type == str:
                config_list.append(config)
            elif config_type == dict:
                configs = _compile_source_configs(**config, rng=rng)
                for config in configs:
                    config_list.append(config)
        expanded_config["sources"][i]["configurations"] = config_list

    return expanded_config


def get_detector_setups(expanded_config: dict) -> list:
    """Permutate the lists of values in the expanded config to
    generate a list of detector setups.

    Args:
        expanded_config: a dictionary representing an expanded seed synthesis configuration

    Returns:
        A list of detector setups
    """
    detector_params = expanded_config["gamma_detector"]["parameters"]
    list_of_parameters_values = [x for x in detector_params.values()]
    parameter_permutations = list(itertools.product(*list_of_parameters_values))
    detector_setups = []
    for perm in parameter_permutations:
        setup = copy.deepcopy(expanded_config["gamma_detector"])
        for i, p in enumerate(detector_params):
            setup["parameters"][p] = perm[i]
        detector_setups.append(setup)

    return detector_setups


def get_inject_setups(config: dict) -> list:
    """Get a list of fully expanded synthesis configurations from an initial,
    collapsed configuration.

    Args:
        config: a dictionary representing a collapsed seed synthesis configuration

    Returns:
        A list of expanded configurations
    """
    expanded_config = get_expanded_config(config)
    detector_setups = get_detector_setups(expanded_config)
    inject_setups = []
    for detector_setup in detector_setups:
        inject_setup = {
            "gamma_detector": detector_setup,
            "sources": expanded_config["sources"]
        }
        inject_setups.append(inject_setup)

    return inject_setups


class GadrasNotInstalledError(Exception):
    """PyRIID could not find a GADRAS installation."""
    pass
