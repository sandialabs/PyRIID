# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic seeds from GADRAS."""
import yaml

from riid.data.synthetic.seed import SeedSynthesizer

seed_synth_config = """
---
detector:
  name: Generic\\NaI\\2x4x16
  distance_cm: 100
  height_cm: 50
  dead_time_us: 5
  elevation_m: 1620
  latitude_deg: 35.0
  longitude_deg: 235.0
foregrounds:
  - isotope: U235
    sources:
      - U235,100uC
      - U235,10uC
  - isotope: Am241
    sources:
      - Am241,100uC
backgrounds:
  - cosmic: true
    terrestrial: true
    K40_percent: 1.56
    U_ppm: 2.11
    Th232_ppm: 5.59
    attenuation: 0
    low_energy_continuum: 0
    high_energy_continuum: 0
    suppression_scalar: 1
  - cosmic: true
    terrestrial: true
    K40_percent: 15.0
    U_ppm: 0.0
    Th232_ppm: 0.0
    attenuation: 0
    low_energy_continuum: 0
    high_energy_continuum: 0
    suppression_scalar: 1
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
#   seeds_ss.to_hdf("seeds.h5")
# or use them with the StaticSynthesizer
