# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This modules contains utilities for generating synthetic gamma spectrum templates from GADRAS."""
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from riid.data import SampleSet
from riid.data.labeling import BACKGROUND_LABEL
from riid.gadras import pcf_to_smpl

if sys.platform == "win32":
    import clr
    import yaml

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
    GADRAS_API_SEEMINGLY_AVAILABLE = os.path.exists(GADRAS_INSTALL_PATH) and \
        os.path.exists(GADRAS_API_CONFIG_FILE_PATH)

    if GADRAS_API_SEEMINGLY_AVAILABLE:
        sys.path.append(GADRAS_ASSEMBLY_PATH)
        clr.AddReference("Sandia.Gadras.API")
        clr.AddReference("Sandia.Gadras.Utilities")
        clr.AddReference("System.Collections")

        from Sandia.Gadras.API import GadrasAPIWrapper, LocationInfo  # noqa
        from Sandia.Gadras.API.Inject import InjectSetup  # noqa
        from Sandia.Gadras.Utilities import Configs  # noqa
        from System.Collections.Generic import List  # noqa


class SeedSynthesizer():
    """ Synthesizes total counts-normalized gamma spectra using the GADRAS API, the purpose of
        which is to seed other synthesizers.
    """

    def __init__(self):
        self.fg_records = 1
        self.bg_records = 1

    @contextmanager
    def _cwd(self, path):
        """ Temporarily changes working directory.

            This is used to change the execution location which necessary due to how the GADRAS API
            uses relative pathing from its installation directory.
        """
        oldpwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(oldpwd)

    def _get_gadras_api(self, instance_num=1, initialize_transport=True):
        """ Sets up and initializes the GADRAS API object.

            Please refer to GADRAS API documentation to learn more.
            Default GADRAS installations currently put these docs here:
            C:\\GADRAS\\Program\\Documentation\\api-docs
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

    def _get_inject_setup(self, gadras_api, output_path, title, record_number, source,
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
        """ Builds a GADRAS InjectSetup.

            Please refer to GADRAS API documentation to learn more.
            Default GADRAS installations currently put these docs here:
            C:\\GADRAS\\Program\\Documentation\\api-docs
        """
        setup = InjectSetup()
        setup.SetDefaults(gadras_api)
        setup.FileName = output_path
        setup.Title = title
        setup.Record = record_number
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

    def _get_foreground_inject_setups(self, gadras_api, detector, sources, isotope, output_path):
        """ Obtains foreground InjectSetups.

            Please refer to GADRAS API documentation to learn more.
            Default GADRAS installations currently put these docs here:
            C:\\GADRAS\\Program\\Documentation\\api-docs
        """
        setups = List[InjectSetup]()
        if not sources:
            return setups
        for source in sources:
            setup = self._get_inject_setup(
                gadras_api=gadras_api,
                output_path=output_path,
                title=isotope,
                record_number=self.fg_records,
                source=source,
                detector_distance_to_source_cm=detector["distance_cm"],
                detector_height_cm=detector["height_cm"],
                detector_dead_time_usecs=detector["dead_time_us"],
                detector_elevation_m=detector["elevation_m"],
                detector_latitude_deg=detector["latitude_deg"],
                detector_longitude_deg=detector["longitude_deg"],
                detector_contains_internal_source=False,  # TODO
            )
            setups.Add(setup)
            self.fg_records += 1
        return setups

    def _get_background_inject_setups(self, gadras_api, detector, sources, output_path):
        """ Obtains background InjectSetups.

            Please refer to GADRAS API documentation to learn more.
            Default GADRAS installations currently put these docs here:
            C:\\GADRAS\\Program\\Documentation\\api-docs
        """
        setups = List[InjectSetup]()
        if not sources:
            return setups
        for source in sources:
            setup = self._get_inject_setup(
                gadras_api=gadras_api,
                output_path=output_path,
                title=BACKGROUND_LABEL,
                record_number=self.bg_records,
                source=None,  # None because we're making background only
                detector_distance_to_source_cm=detector["distance_cm"],
                detector_height_cm=detector["height_cm"],
                detector_dead_time_usecs=detector["dead_time_us"],
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
            self.bg_records += 1
        return setups

    def generate(self, config: Union[str, dict], normalize_sources=True,
                 verbose: bool = False) -> SampleSet:
        """Produces a SampleSet containing foreground and/or background seeds using GADRAS based
        on the given inject configuration.

        Args:
            config: a dictionary is treated as the actual config containing the needed information
                to perform injects via the GADRAS API, while a string is treated as a path to a YAML
                file which deserialized as a dictionary.
            normalize_sources: Whether to divide each row of the SampleSet's sources
                DataFrame by its sum. Defaults to True.
            verbose: when True, displays extra output.

        Returns:
            A SampleSet containing foreground and/or background seeds generated by GADRAS.
        """

        if not GADRAS_API_SEEMINGLY_AVAILABLE:
            msg = "GADRAS API not found; no injects can be performed."
            raise GadrasNotInstalledError(msg)

        if isinstance(config, str):
            with open(config, "r") as stream:
                config = yaml.safe_load(stream)
        elif not isinstance(config, dict):
            msg = (
                "The provided config for seed synthesis must either be "
                "a path to a properly structured YAML file or "
                "a properly structured dictionary."
            )
            raise ValueError(msg)

        seeds_ss = SampleSet()
        with self._cwd(GADRAS_ASSEMBLY_PATH):
            gadras_api = self._get_gadras_api()
            detector_name = config["detector"]["name"]
            gadras_api.detectorSetCurrent(detector_name)
            now = datetime.utcnow().isoformat().replace(":", "_")
            worker = gadras_api.GetBatchInjectWorker()

            # Generate foreground seeds
            rel_fg_output_path = f"{now}_fg.pcf"
            abs_fg_output_path = os.path.join(
                GADRAS_DETECTOR_DIR_PATH,
                detector_name,
                rel_fg_output_path
            )
            fg_injects_exist = False
            for fg in config["foregrounds"]:
                fg_inject_setups = self._get_foreground_inject_setups(
                    gadras_api,
                    config["detector"],
                    fg["sources"],
                    fg["isotope"],
                    rel_fg_output_path
                )
                if fg_inject_setups:
                    worker.Run(fg_inject_setups)
                    fg_injects_exist = True
            if fg_injects_exist:
                # Add source name to foreground seeds file
                fg_records = 1
                for fg in config["foregrounds"]:
                    for source in fg["sources"]:
                        gadras_api.spectraFileWriteSingleField(
                            rel_fg_output_path,
                            fg_records,
                            "Source",
                            source
                        )
                        fg_records += 1
                if verbose:
                    logging.info(f"Foreground seeds saved to: {abs_fg_output_path}")
                fg_ss = pcf_to_smpl(abs_fg_output_path)
                seeds_ss.concat(fg_ss)

            # Generate background seeds
            rel_bg_output_path = f"{now}_bg.pcf"
            abs_bg_output_path = os.path.join(
                GADRAS_DETECTOR_DIR_PATH,
                detector_name,
                rel_bg_output_path
            )
            bg_inject_setups = self._get_background_inject_setups(
                gadras_api,
                config["detector"],
                config["backgrounds"],
                rel_bg_output_path
            )
            if bg_inject_setups:
                worker.Run(bg_inject_setups)
                # Add custom-built source name to background seeds file
                bg_records = 1
                for bg in config["backgrounds"]:
                    k_percent, u_ppm, th_ppm = \
                        bg["K40_percent"], bg["U_ppm"], bg["Th232_ppm"]
                    cosmic = "+cosmic" if bg["cosmic"] else ""
                    source = f"K={k_percent}%,U={u_ppm}ppm,Th={th_ppm}ppm{cosmic}"
                    gadras_api.spectraFileWriteSingleField(
                        rel_bg_output_path,
                        bg_records,
                        "Source",
                        source
                    )
                    bg_records += 1
                if verbose:
                    logging.info(f"Background seeds saved to: {abs_bg_output_path}")
                bg_ss = pcf_to_smpl(abs_bg_output_path)
                seeds_ss.concat(bg_ss)

        seeds_ss.normalize(p=1)
        if normalize_sources:
            seeds_ss.normalize_sources()
        seeds_ss.detector_info = config["detector"]

        return seeds_ss


class SeedMixer():
    def __init__(self, mixture_size: int = 2, min_source_contribution: float = 0.1):
        assert mixture_size >= 2
        assert min_source_contribution >= 0.1
        assert mixture_size * min_source_contribution < 1.0

        self.mixture_size = mixture_size
        self.min_source_contribution = min_source_contribution

    def generate(self, seeds_ss: SampleSet, n_samples: int = 10000) -> SampleSet:
        """Computes random mixtures of seeds across the isotope level.

            n_mixture = seed_1 * ratio_1 + seed_2 * ratio_2 + ... + seed_n * ratio_n
                where:
                - ratio_1 + ratio_2 + ... + ratio_n = 1
                - sum(seed_i) = 1
                - sum(n_mixture) = self.mixture_size
                  (this is before re-normalizing, at which point it will sum to 1)

            For 3 contributors:
                running_contribution = 0.0
                contribution1 = uniform(min_contribution, 1 - running_contribution
                                - n_remaining_contributors * min_contribution)
                              = uniform(0.1, 1 - 0.0 - 2 * 0.1)
                              = uniform(0.1, 0.8)
                              = 0.80
                running_contribution += contribution1
                contribution2 = uniform(0.1, 1 - running_contribution - 1 * 0.1)
                              = uniform(0.1, 0.1)
                              = 0.10
                running_contribution += contribution2
                contribution3 = 1 - running_contribution
        """
        if seeds_ss and not seeds_ss.all_spectra_sum_to_one():
            raise ValueError("At least one provided seed does not sum close to 1.")

        if not np.all(np.count_nonzero(seeds_ss.get_source_contributions().values, axis=1) == 1):
            raise ValueError("At least one provided seed contains mixture of sources.")

        for ecal_column in seeds_ss.ECAL_INFO_COLUMNS:
            if not np.all(np.isclose(seeds_ss.info[ecal_column], seeds_ss.info[ecal_column][0])):
                raise ValueError("At least one ecal value is different than the others.")

        non_bg_seeds_ss = seeds_ss[seeds_ss.get_labels() != BACKGROUND_LABEL]
        non_bg_seeds_ss.sources.drop(
            BACKGROUND_LABEL,
            axis=1,
            level="Isotope",
            inplace=True,
            errors='ignore'
        )
        isotopes = non_bg_seeds_ss.get_labels().values
        n_sources = non_bg_seeds_ss.n_samples
        unique_isotopes, indices = np.unique(isotopes, return_index=True)
        n_isotopes = len(unique_isotopes)

        # preserve original order of isotopes (np.unique() sorts them)
        unique_isotopes = np.array([isotopes[i] for i in sorted(indices)])
        cnts = np.array([np.count_nonzero(isotopes == isotope) for isotope in unique_isotopes])

        isotope_inds = [np.arange(cnts[:idx].sum(), cnts[:idx+1].sum())
                        for idx in range(n_isotopes)]
        isotope_dict = dict(zip(unique_isotopes, isotope_inds))
        mixture_inds = {i+1: [] for i in range(self.mixture_size)}
        mixture_inds[1] = [(each,) for each in range(n_sources)]

        # first generate mixture indices
        for n in range(2, self.mixture_size+1):
            for mix_idx in mixture_inds[n-1]:
                # get first source index for next isotope after last isotope in the mixture
                last_isotope = isotopes[mix_idx[-1]]
                if last_isotope != unique_isotopes[-1]:
                    next_source = np.where(unique_isotopes == last_isotope)[0].squeeze() + 1
                    next_isotope = unique_isotopes[next_source]
                    next_idx = isotope_dict[next_isotope][0]

                    # generate next set of mixture indices
                    mixtures = [(*mix_idx, each) for each in range(next_idx, n_sources)]
                    mixture_inds[n].extend(mixtures)

        # generate sampling probability distribution to accomadate mixture balancing
        # at isotope level
        flat_mixture_inds = [item for sublist in mixture_inds[self.mixture_size]
                             for item in sublist]
        flat_mixture_sources = [isotopes[each] for each in flat_mixture_inds]
        unique_isotopes_sorted, isotope_occurences = np.unique(flat_mixture_sources,
                                                               return_counts=True)
        isotope_weights = 1/isotope_occurences
        isotope_weights_dict = dict(zip(unique_isotopes_sorted, isotope_weights))

        mixture_weights = np.zeros(len(mixture_inds[self.mixture_size]))
        for idx, mixture in enumerate(mixture_inds[self.mixture_size]):
            mixture_weights[idx] = sum([isotope_weights_dict[isotopes[ind]] for ind in mixture])
        mixture_weights = mixture_weights/mixture_weights.sum()  # normalize to make pdf

        # randomly sample mixtures
        random_seed_inds = np.random.choice(len(mixture_weights),
                                            size=n_samples,
                                            replace=True,
                                            p=mixture_weights)
        random_isotopes = list(sum([mixture_inds[self.mixture_size][each]
                               for each in random_seed_inds], ()))
        random_isotopes = [isotopes[each] for each in random_isotopes]

        mixture_seeds = np.zeros((n_samples, seeds_ss.n_channels))
        source_matrix = np.zeros((n_samples, n_sources))

        # create mixture seeds
        for idx, random_mixture_ind in enumerate(random_seed_inds):
            # randomly sample probability distribution
            ratios = [0.0 for i in range(self.mixture_size)]
            for ratio_idx in range(self.mixture_size - 1):
                ratios[ratio_idx] = np.random.uniform(self.min_source_contribution,
                                                      1 - sum(ratios) -
                                                      (self.mixture_size - ratio_idx + 1) *
                                                      self.min_source_contribution)
            ratios[-1] = 1.0 - sum(ratios)

            # generate mixture data
            mixture = mixture_inds[self.mixture_size][random_mixture_ind]
            source_contributions = np.zeros(n_sources)
            for ratio_idx, spectra_idx in enumerate(mixture):
                source_contribution = seeds_ss.spectra.values[spectra_idx, :]\
                    * ratios[ratio_idx]
                mixture_seeds[idx, :] += source_contribution
                source_contributions[spectra_idx] = ratios[ratio_idx]
            source_matrix[idx, :] = source_contributions

        source_matrix = np.array(source_matrix)

        mixture_ss = SampleSet()
        mixture_ss.spectra = pd.DataFrame(
            mixture_seeds
        )
        mixture_ss.sources = pd.DataFrame(
            source_matrix,
            columns=non_bg_seeds_ss.sources.columns
        )

        # populate SampleSet info
        mixture_ss.info = pd.DataFrame(
            np.full((mixture_ss.spectra.shape[0], seeds_ss.info.shape[1]), None),
            columns=seeds_ss.info.columns
        )

        for ecal_column in seeds_ss.ECAL_INFO_COLUMNS:
            mixture_ss.info.loc[:, ecal_column] = seeds_ss.info[ecal_column][0]
        mixture_ss.info.loc[:, 'pyriid_version'] = seeds_ss.info['pyriid_version'][0]
        mixture_ss.info.loc[:, 'tag'] = seeds_ss.info['tag'][0]

        # TODO: fill in rest of columns
        # description, timestamp, live_time, real_time, snr_target, snr_estimate, sigma, bg_counts,
        # fg_counts, bg_counts_expected, fg_counts_expected, total_counts, total_neutron_counts,
        # distance_cm, area_density, atomic_number

        return mixture_ss


class GadrasNotInstalledError(Exception):
    pass
