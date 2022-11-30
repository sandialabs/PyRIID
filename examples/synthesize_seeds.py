# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic seeds from GADRAS."""
import yaml

from riid.data.synthetic.seed import SeedSynthesizer

seed_synth_config = """
---
gamma_detector:
  name: 3x3\\NaI MidScat
  parameters:
    distance_cm: 100
    height_cm: 10
    dead_time_per_pulse: 5
    latitude_deg: 35.0
    longitude_deg: 253.4
    elevation_m: 1620
sources:
  - isotope: Am241
    configurations:
      - Am241,100uC
  - isotope: Ba133
    configurations:
      - Ba133,100uC
  - isotope: Cs137
    configurations:
      - Cs137,100uC
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

seed_synth = SeedSynthesizer()
seeds_ss = seed_synth.generate(
    seed_synth_config,
    verbose=True
)

print(seeds_ss)

# At this point, you could save out the seeds via:
seeds_ss.to_hdf("seeds.h5")

# or start separating your backgrounds from foreground for use with the StaticSynthesizer
seed_labels = seeds_ss.get_labels(target_level="Seed")
bg_seed_names = (
    "Cosmic",
    "PotassiumInSoil",
    "UraniumInSoil",
    "ThoriumInSoil",
)
bg_indices = seed_labels.isin(bg_seed_names)

fg_seeds_ss = seeds_ss[~bg_indices]
bg_seeds_ss = seeds_ss[bg_indices]

print(fg_seeds_ss)
print(bg_seeds_ss)

fg_seeds_ss.to_hdf("./fg_seeds.h5")
bg_seeds_ss.to_hdf("./bg_seeds.h5")
