# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for working with the GADRAS API."""
import sys

GADRAS_API_SEEMINGLY_AVAILABLE = False

if sys.platform == "win32":
    import json
    import logging
    import os

    import clr
    from jsonschema import validate
    from riid.data import SampleSet
    from riid.data.labeling import BACKGROUND_LABEL
    from riid.gadras.pcf import pcf_to_smpl

    GADRAS_INSTALL_PATH = "C:\\GADRAS"
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
    if GADRAS_API_SEEMINGLY_AVAILABLE:
        sys.path.append(GADRAS_ASSEMBLY_PATH)
        clr.AddReference("Sandia.Gadras.API")
        clr.AddReference("Sandia.Gadras.Utilities")
        clr.AddReference("System.Collections")

        from Sandia.Gadras.API import GadrasAPIWrapper, LocationInfo  # noqa
        from Sandia.Gadras.API.Inject import InjectSetup  # noqa
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
    "DET_SETBACK": {"type": "float"},
    # "HEIGHT_CM": {"type": "float"},  # changed in inject setup instead
    # "EFFICIENCY_HOLDER": {"type": "float"},
    "ECAL_ORD_0": {"type": "float"},
    "ECAL_ORD_1": {"type": "float"},
    "ECAL_ORD_2": {"type": "float"},
    "ECAL_ORD_3": {"type": "float"},
    "ECAL_LOW_E": {"type": "float"},
    # "ECAL_ORD_0_VARYING": {"type": "int"},
    # "ECAL_ORD_1_VARYING": {"type": "int"},
    # "ECAL_ORD_2_VARYING": {"type": "int"},
    # "ECAL_ORD_3_VARYING": {"type": "int"},
    # "ECAL_LOW_E_VARYING": {"type": "int"},
    "ERES_OFFSET": {"type": "float"},
    "ERES_FWHM": {"type": "float"},
    "ERES_POWER": {"type": "float"},
    "ERES_LOW_ESKEW": {"type": "float"},
    "LOW_ERES_SKEW_EXTENT": {"type": "float"},
    "HIGH_ERES_SKEW_EXTENT": {"type": "float"},
    "ERES_HIGH_ESKEW": {"type": "float"},
    "LOW_ERES_SKEW_POWER": {"type": "float"},
    "HIGH_ERES_SKEW_POWER": {"type": "float"},
    # "ERES_OFFSET_VARYING": {"type": "int"},
    # "ERES_FWHM_VARYING": {"type": "int"},
    # "ERES_POWER_VARYING": {"type": "int"},
    # "ERES_LOW_ESKEW_VARYING": {"type": "int"},
    # "LOW_ERES_SKEW_POWER_VARYING": {"type": "int"},
    # "HIGH_ERES_SKEW_POWER_VARYING": {"type": "int"},
    # "ERES_HIGH_ESKEW_VARYING": {"type": "int"},
    # "LOW_ERES_SKEW_EXTENT_VARYING": {"type": "int"},
    # "HIGH_ERES_SKEW_EXTENT_VARYING": {"type": "int"},
    # "SOLID_ANGLE_PERCENT": {"type": "float"},
    "DET_LENGTH_CM": {"type": "float"},
    "DET_WIDTH_CM": {"type": "float"},
    # "DET_HEIGHT_WIDTH_RATIO": {"type": "float"},
    # "DET_HEIGHT_WIDTH_RATIO_VARYING": {"type": "int"},
    # "SHAPE_FACTOR": {"type": "float"},
    # "EFF_SCALAR": {"type": "float"},
    # "DET_LENGTH_CM_VARYING": {"type": "int"},
    # "DET_WIDTH_CM_VARYING": {"type": "int"},
    # "SHAPE_FACTOR_VARYING": {"type": "int"},
    # "EFF_SCALAR_VARYING": {"type": "int"},
    "SCATTER_ATTENUATE": {"type": "float"},
    "SCATTER_CLUTTER": {"type": "float"},
    "SCATTER_PROB_AT_0": {"type": "float"},
    "SCATTER_PROB_AT_45": {"type": "float"},
    "SCATTER_PROB_AT_90": {"type": "float"},
    "SCATTER_PROB_AT_135": {"type": "float"},
    "SCATTER_PROB_AT_180": {"type": "float"},
    "SCATTER_PREF_ANGLE": {"type": "float"},
    "SCATTER_PREF_ANGLE_DELTA": {"type": "float"},
    "SCATTER_PREF_ANGLE_MAG": {"type": "float"},
    "SCATTER_FLATTEN_EDGE": {"type": "float"},
    "SCATTER_RATE_TOWARDS_ZERO": {"type": "float"},
    "SCATTER_INCREASE_WITH_E": {"type": "float"},
    # "SCATTER_ATTENUATE_VARYING": {"type": "int"},
    # "SCATTER_CLUTTER_VARYING": {"type": "int"},
    # "SCATTER_PROB_AT_0_VARYING": {"type": "int"},
    # "SCATTER_PROB_AT_45_VARYING": {"type": "int"},
    # "SCATTER_PROB_AT_90_VARYING": {"type": "int"},
    # "SCATTER_PROB_AT_135_VARYING": {"type": "int"},
    # "SCATTER_PROB_AT_180_VARYING": {"type": "int"},
    # "SCATTER_PREF_ANGLE_VARYING": {"type": "int"},
    # "SCATTER_PREF_ANGLE_DELTA_VARYING": {"type": "int"},
    # "SCATTER_PREF_ANGLE_MAG_VARYING": {"type": "int"},
    # "SCATTER_FLATTEN_EDGE_VARYING": {"type": "int"},
    # "SCATTER_RATE_TOWARDS_ZERO_VARYING": {"type": "int"},
    # "SCATTER_INCREASE_WITH_E_VARYING": {"type": "int"},
    # "SHIELD_ANGLE_PERCENT": {"type": "float"},
    # "SHIELD_THICKNESS_CM": {"type": "float"},
    # "DEAD_LAYER_Z": {"type": "float"},
    # "DEAD_LAYER_G_CM2": {"type": "float"},
    # "SHIELD_LLD_KEV": {"type": "float"},
    # "SHIELD_MATERIAL_ID": {"type": "int"},
    # "SHIELD_ANGLE_PERCENT_VARYING": {"type": "int"},
    # "SHIELD_THICKNESS_CM_VARYING": {"type": "int"},
    # "DEAD_LAYER_Z_VARYING": {"type": "int"},
    # "DEAD_LAYER_G_CM2_VARYING": {"type": "int"},
    # "SHIELD_LLD_KEV_VARYING": {"type": "int"},
    # "EXT_ANNIHILATION": {"type": "float"},
    # "SHAPE_TIME": {"type": "float"},
    # "XRAY_SOURCE_ID": {"type": "int"},
    # "XRAY_SOURCE2_ID": {"type": "int"},
    # "BETAS": {"type": "float"},
    # "HOLE_MU_TAU": {"type": "float"},
    # "DEAD_LAYER": {"type": "float"},
    # "LLD_KEV": {"type": "float"},
    # "LLD_SHARPNESS": {"type": "float"},
    # "FRISCH_GRID": {"type": "float"},
    # "LOCAL_XRAY_MAG": {"type": "float"},
    # "LOCAL_XRAY2_MAG": {"type": "float"},
    # "EXT_ANNIHILATION_VARYING": {"type": "int"},
    # "SHAPE_TIME_VARYING": {"type": "int"},
    # "BETAS_VARYING": {"type": "int"},
    # "HOLE_MU_TAU_VARYING": {"type": "int"},
    # "DEAD_LAYER_VARYING": {"type": "int"},
    # "LLD_KEV_VARYING": {"type": "int"},
    # "LLD_SHARPNESS_VARYING": {"type": "int"},
    # "FRISCH_GRID_VARYING": {"type": "int"},
    # "DETECTOR_TYPE": {"type": "int"},
    # "DEFAULT_CHANNELS": {"type": "int"},
    # "MIN_ENERGY_KEV": {"type": "int"},
    # "MAX_ENERGY_KEV": {"type": "int"},
    # "INBIN": {"type": "int"},
    # "REBIN": {"type": "int"},
    # "PILEUP": {"type": "int"},
    # "INNER_AN": {"type": "float"},
    # "INNER_AD": {"type": "float"},
    # "INNER_POROSITY": {"type": "float"},
    # "INNER_AN_VARYING": {"type": "int"},
    # "INNER_AD_VARYING": {"type": "int"},
    # "INNER_POROSITY_VARYING": {"type": "int"},
    # "OUTER_AN": {"type": "float"},
    # "OUTER_AD": {"type": "float"},
    # "OUTER_POROSITY": {"type": "float"},
    # "OUTER_AN_VARYING": {"type": "int"},
    # "OUTER_AD_VARYING": {"type": "int"},
    # "OUTER_POROSITY_VARYING": {"type": "int"},
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
    # "AIR_PRESSURE": {"type": "float"},
    # "AC_SHIELD_COMPTON_CAMERA": {"type": "int"},
    # "ELEVATION": {"type": "float"},  # changed in inject setup instead
    # "NEUTRON_SHIELD_PERC": {"type": "float"},
    # "NEUTRON_REFLECT": {"type": "float"},
    # "NEUTRON_ENVIRONMENT": {"type": "float"},
    # "NEUTRON_SHIELD_PERC_VARYING": {"type": "int"},
    # "NEUTRON_REFLECT_VARYING": {"type": "int"},
    # "TEMPLATE_ERROR": {"type": "float"},
    # "SCATTER_ENVIRONMENT": {"type": "float"},
    # "DEAD_TIME_PER_PULSE": {"type": "float"},  # changed in inject setup instead
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
    """Synthesizes total counts-normalized source gamma spectra using the GADRAS API,
        the purpose of which is to seed other synthesizers.
    """

    def __init__(self, gadras_api=None) -> None:
        super().__init__(gadras_api)

    def _get_inject_setups_for_sources(self, gadras_api, detector, sources, isotope, output_path):
        """Obtains InjectSetups for sources.
        """
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
        """Produces a SampleSet containing foreground and/or background seeds using GADRAS based
        on the given inject configuration.

        Args:
            config: a dictionary containing the needed information to perform injects
                via the GADRAS API
            verbose: when True, displays extra output.

        Returns:
            A SampleSet containing foreground and/or background seeds generated by GADRAS.
        """
        worker = self.gadras_api.GetBatchInjectWorker()

        injects_exist = False
        for fg in config["foregrounds"]:
            inject_setups = self._get_inject_setups_for_sources(
                self.gadras_api,
                config["gamma_detector"]["parameters"],
                fg["sources"],
                fg["isotope"],
                rel_output_path
            )
            if inject_setups:
                worker.Run(inject_setups)
                injects_exist = True

        if not injects_exist:
            return

        # Add source name to foreground seeds file
        record_num = 1
        for fg in config["foregrounds"]:
            for source in fg["sources"]:
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
            logging.info(f"Foreground seeds saved to: {abs_output_path}")

        return abs_output_path


class BackgroundInjector(BaseInjector):
    """ Synthesizes total counts-normalized background gamma spectra using the GADRAS API,
        the purpose of which is to seed other synthesizers.
    """

    def __init__(self, gadras_api: GadrasAPIWrapper = None) -> None:
        super().__init__(gadras_api)

    def _get_inject_setups_for_backgrounds(self, gadras_api, detector, backgrounds, output_path):
        """Obtains InjectSetups for background seeds.
        """
        setups = List[InjectSetup]()
        if not backgrounds:
            return setups
        for i, source in enumerate(backgrounds, start=self.n_records):
            setup = get_inject_setup(
                gadras_api=gadras_api,
                output_path=output_path,
                title=BACKGROUND_LABEL,
                record_num=i,
                source=None,  # None because we're making background only
                detector_distance_to_source_cm=detector["distance_cm"],
                detector_height_cm=detector["height_cm"],
                detector_dead_time_usecs=detector["dead_time_per_pulse"],
                detector_elevation_m=detector["elevation_m"],
                detector_latitude_deg=detector["latitude_deg"],
                detector_longitude_deg=detector["longitude_deg"],
                detector_contains_internal_source=False,  # TODO
                background_include_cosmic=source["cosmic"],
                background_include_terrestrial=source["terrestrial"],
                background_K40_percent=source["K40_percent"],
                background_U_ppm=source["U_ppm"],
                background_Th232_ppm=source["Th232_ppm"],
                background_low_energy_continuum=source["low_energy_continuum"],
                background_high_energy_continuum=source["high_energy_continuum"],
                background_attenuation=source["attenuation"],
                background_suppression_scalar=source["suppression_scalar"]
            )
            setups.Add(setup)
            self.n_records += 1
        return setups

    def generate(self, config: dict, rel_output_path: str, verbose: bool = False) -> SampleSet:
        """Produces a SampleSet containing foreground and/or background seeds using GADRAS based
        on the given inject configuration.

        Args:
            config: a dictionary is treated as the actual config containing the needed information
                to perform injects via the GADRAS API, while a string is treated as a path to a YAML
                file which deserialized as a dictionary.
            verbose: when True, displays extra output.

        Returns:
            A SampleSet containing foreground and/or background seeds generated by GADRAS.
        """
        worker = self.gadras_api.GetBatchInjectWorker()

        inject_setups = self._get_inject_setups_for_backgrounds(
            self.gadras_api,
            config["gamma_detector"]["parameters"],
            config["backgrounds"],
            rel_output_path,
        )

        if not inject_setups:
            return

        worker.Run(inject_setups)

        # Add custom-built source name to background seeds file
        record_num = 1
        for bg in config["backgrounds"]:
            k_percent, u_ppm, th_ppm = \
                bg["K40_percent"], bg["U_ppm"], bg["Th232_ppm"]
            cosmic = "+cosmic" if bg["cosmic"] else ""
            source = f"K={k_percent}%,U={u_ppm}ppm,Th={th_ppm}ppm{cosmic}"
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
            logging.info(f"Background seeds saved to: {abs_output_path}")

        return abs_output_path


def get_gadras_api(instance_num=1, initialize_transport=True):
    """Sets up and initializes the GADRAS API object.
    """
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
    """Builds a GADRAS InjectSetup object.
    """
    setup = InjectSetup()
    setup.SetDefaults(gadras_api)
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
    setup.UpdateBackgroundLocation(loc_info)
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


def get_counts_per_bg_source_unit(gadras_api, detector, worker, bg_source):
    BACKGROUND_SOURCES = ("K", "U", "T", "Cosmic")
    if bg_source not in BACKGROUND_SOURCES:
        msg = (
            f"{bg_source} is invalid. "
            f"Acceptable options are: {BACKGROUND_SOURCES}"
        )
        raise ValueError(msg)

    rel_output_path = f"{bg_source}_Background.pcf"
    inject_setup = get_inject_setup(
        gadras_api=gadras_api,
        output_path=rel_output_path,
        title=bg_source,
        record_num=1,
        source=None,
        detector_distance_to_source_cm=detector["distance_cm"],
        detector_height_cm=detector["height_cm"],
        detector_dead_time_usecs=detector["dead_time_per_pulse"],
        detector_elevation_m=detector["elevation_m"],
        detector_latitude_deg=detector["latitude_deg"],
        detector_longitude_deg=detector["longitude_deg"],
        background_include_terrestrial=True,
        background_K40_percent=1,
    )
    setups = List[InjectSetup]()
    setups.Add(inject_setup)
    worker.Run(setups)
    ss = pcf_to_smpl(rel_output_path)
    os.remove(rel_output_path)
    counts_per_unit = ss.spectra.iloc[0].sum()
    return counts_per_unit


def validate_inject_config(config: dict):
    validate(instance=config, schema=GADRAS_API_SCHEMA)


class GadrasNotInstalledError(Exception):
    pass
