# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates additional ways to generate synthetic seeds from GADRAS:
- Example 1: inject everything in a folder ending in .gam
- Example 2: build and inject point sources comprised of multiple radioisotopes
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from riid import SeedSynthesizer
from riid.gadras.api import GADRAS_INSTALL_PATH


def convert_df_row_to_inject_string(row):
    """Converts a row of the DataFrame to a proper GADRAS inject string"""
    isotopes = row.index.values
    activities = row.values
    isotopes_and_activities = [
        f"{iso},{round(act, 4)}uCi" for iso, act in zip(isotopes, activities)
    ]  # Note the "uCi" specified here. You may need to change it.
    inject_string = " + ".join(isotopes_and_activities)
    return inject_string


seed_synth_config = """
---
gamma_detector:
  name: Generic\\NaI\\3x3\\Front\\MidScat
  parameters:
    distance_cm: 100
    height_cm: 10
    dead_time_per_pulse: 5
    latitude_deg: 35.0
    longitude_deg: 253.4
    elevation_m: 1620
sources:
  - isotope: U235
    configurations: null
...
"""
seed_synth_config = yaml.safe_load(seed_synth_config)

try:
    seed_synth = SeedSynthesizer()

    # Example 1
    # Change "Continuum" to your own source directory
    gam_dir = Path(GADRAS_INSTALL_PATH).joinpath("Source/Continuum")
    gam_filenames = [x.stem for x in gam_dir.glob("*.gam")]
    seed_synth_config["sources"][0]["configurations"] = gam_filenames
    seeds = seed_synth.generate(seed_synth_config)
    seeds.to_hdf("seeds_from_gams.h5")

    # Example 2
    # For the following DataFrame, columns are isotopes, rows are samples, and cells are activity
    df = pd.DataFrame(
        np.random.rand(25, 5),  # Reminder: ensure all activity values are in the same units
        columns=["Am241", "Ba133", "Co60", "Cs137", "U235"],
    )
    configurations = df.apply(convert_df_row_to_inject_string, axis=1).to_list()
    seed_synth_config["sources"][0]["configurations"] = configurations

    seeds = seed_synth.generate(seed_synth_config)
    seeds.to_hdf("seeds_from_df.h5")
except FileNotFoundError:
    pass  # Happens when not on Windows
