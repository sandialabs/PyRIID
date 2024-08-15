# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic seeds from GADRAS using PyRIID's
configuration expansion features."""
import yaml

from riid import SeedSynthesizer

seed_synth_config = """
---
gamma_detector:
  name: Generic\\NaI\\2x4x16
  parameters:
    distance_cm:
      - 10
      - 100
      - 1000
    height_cm: 100
    dead_time_per_pulse: 5
    latitude_deg: 35.0
    longitude_deg: 253.4
    elevation_m: 1620
sources:
  - isotope: Cs137
    configurations:
      - Cs137,100uCi
      - name: Cs137
        activity:
          - 1
          - 0.5
        activity_units: Ci
        shielding_atomic_number:
          min: 10
          max: 40.0
          dist: uniform
          num_samples: 5
        shielding_aerial_density:
          mean: 120
          std: 2
          num_samples: 5
  - isotope: Cosmic
    configurations:
      - Cosmic
  - isotope: K40
    configurations:
      - PotassiumInSoil
  - isotope: Ra226
    configurations:
      - UraniumInSoil
  - isotope: Th232
    configurations:
      - ThoriumInSoil
...
"""
seed_synth_config = yaml.safe_load(seed_synth_config)

try:
    seeds_ss = SeedSynthesizer().generate(
        seed_synth_config,
        verbose=True
    )
    print(seeds_ss)

    # At this point, you could save out the seeds via:
    seeds_ss.to_hdf("seeds.h5")

    # or start separating your backgrounds from foreground for use with the StaticSynthesizer
    fg_seeds_ss, bg_seeds_ss = seeds_ss.split_fg_and_bg()

    print(fg_seeds_ss)
    print(bg_seeds_ss)

    fg_seeds_ss.to_hdf("./fg_seeds.h5")
    bg_seeds_ss.to_hdf("./bg_seeds.h5")
except FileNotFoundError:
    pass  # Happens when not on Windows
