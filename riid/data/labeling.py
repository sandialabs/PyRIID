# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utility functions for managing ground truth information."""
import logging
import re

BACKGROUND_LABEL = "Background"
NO_SEED = "Unknown"
NO_ISOTOPE = "Unknown"
NO_CATEGORY = "Uncategorized"
CATEGORY_ISOTOPES = {
    "Fission Product": {
        "severity": 3,
        "isotopes": [
            "Ag112",
            "As78",
            "Ba139",
            "Ce143",
            "I132",
            "I133",
            "I134",
            "I135",
            "Kr85m",
            "Kr87",
            "La140",
            "La142",
            "Nd149",
            "Pm150",
            "Rh105",
            "Ru105",
            "Sb115",
            "Sb129",
            "Sr91",
            "Sr92",
            "Te132",
            "Y93",
            "Y91m",
            "Zr95",
        ]
    },
    "Industrial": {
        "severity": 2,
        "isotopes": [
            "Am241",
            "Ba133",
            "Ba140",
            "Bi207",
            "Cf249",
            "Cf250",
            "Cf251",
            "Cf252",
            "Cm244",
            "Co57",
            "Co60",
            "Cs137",
            "Eu152",
            "Eu154",
            "H3",
            "Ho166m",
            "Ir192",
            "Na22",
            "P32",
            "P33",
            "Po210",
            "Se75",
            "Sr90",
            "Tc99",
            "Y88",
        ],
    },
    "Medical": {
        "severity": 3,
        "isotopes": [
            "F18",
            "Ga67",
            "Ga68",
            "Ge68",
            "I123",
            "I124",
            "I125",
            "I129",
            "I131",
            "In111",
            "Lu177m",
            "Mo99",
            "Pd103",
            "Ra223",
            "Rb82",
            "Sm153",
            "Tc99m",
            "Tl201",
            "Xe133",
        ],
    },
    "NORM": {
        "severity": 1,
        "isotopes": [
            "Cosmic",
            "K40",
            "Pb210",
            "Ra226",
            "Th232",
        ],
    },
    "SNM": {
        "severity": 4,
        "isotopes": [
            "Np237",
            "Pu238",
            "Pu239",
            "U232",
            "U233",
            "U235",
            "U237",
            # 16,000 years from now, once the chemically separated uranium has equilibrated
            # with Ra226, then we will need to reconsider U238's categorization.
            "U238",
        ],
    },
}
ISOTOPES = sum(
    [c["isotopes"] for c in CATEGORY_ISOTOPES.values()],
    []
)  # Concatenating the lists of isotopes into one list
SEED_TO_ISOTOPE_SPECIAL_CASES = {
    "ThPlate": "Th232",
    "ThPlate+Thxray,10uC": "Th232",
    "fiestaware": "U238",
    "Uxray,100uC": "U238",
    "DUOxide": "U238",
    "ShieldedDU": "U238",
    "modified_berpball": "Pu239",
    "10 yr WGPu in Fe": "Pu239",
    "1gPuWG_0.5yr,3{an=10,ad=5}": "Pu239",
    "pu239_1yr": "Pu239",
    "pu239_5yr": "Pu239",
    "pu239_10yr": "Pu239",
    "pu239_25yr": "Pu239",
    "pu239_50yr": "Pu239",
    "1kg HEU + 800uCi Cs137": "U235",
    "WGPu + Cs137": "Pu239",
    "HEU": "U235",
    "DU": "U238",
    "WGPu": "Pu239",
    "PuWG": "Pu239",
    "RTG": "Pu238",
    "PotassiumInSoil": "K40",
    "UraniumInSoil": "Ra226",
    "ThoriumInSoil": "Th232",
    "Cosmic": "Cosmic",
}


def _find_isotope(seed: str, verbose=True):
    """Attempt to find the category for the given seed.

    Args:
        seed: string containing the isotope name
        verbose: whether log warnings

    Returns:
        Isotope if found, otherwise NO_ISOTOPE
    """
    if seed.lower() == BACKGROUND_LABEL.lower():
        return BACKGROUND_LABEL
    if seed == NO_ISOTOPE:
        return NO_ISOTOPE

    isotopes = []
    for i in ISOTOPES:
        if i in seed:
            isotopes.append(i)

    n_isotopes = len(isotopes)
    if n_isotopes > 1:
        # Use the longest matching isotope (handles sources strings for things like Tc99 vs Tc99m)
        chosen_match = max(isotopes)
        if verbose:
            logging.warning((
                f"Found multiple isotopes whose names are subsets of '{seed}';"
                f" '{chosen_match}' was chosen."
            ))
        return chosen_match
    elif n_isotopes == 0:
        return NO_ISOTOPE
    else:
        return isotopes[0]


def _find_category(isotope: str):
    """Attempt to find the category for the given isotope.

    Args:
        isotope: string containing the isotope name

    Returns:
        Category if found, otherwise NO_CATEGORY
    """
    if isotope.lower() == BACKGROUND_LABEL.lower():
        return BACKGROUND_LABEL
    if isotope == NO_CATEGORY:
        return NO_CATEGORY

    categories = []
    for c, v in CATEGORY_ISOTOPES.items():
        c_severity = v["severity"]
        c_isotopes = v["isotopes"]
        for i in c_isotopes:
            if i in isotope:
                categories.append((c, c_severity))

    n_categories = len(categories)
    if n_categories > 1:
        return max(categories, key=lambda x: x[1])[0]
    elif n_categories == 0:
        return NO_CATEGORY
    else:
        return categories[0][0]


def label_to_index_element(label_val: str, label_level="Isotope", verbose=False) -> tuple:
    """Try to map a label to a tuple for use in `DataFrame` `MultiIndex` columns.

    Depending on the level of the label value, you will get different tuple:

    | Label Level  | Resulting Tuple          |
    |:-------------|:-------------------------|
    |Seed          |(Category, Isotope, Seed) |
    |Isotope       |(Category, Isotope)       |
    |Category      |(Category,)               |

    Args:
        label_val: part of the label (Category, Isotope, Seed)
            from which to map the other two label values, if possible
        label_level: level of the part of the label provided, e.g,
            "Category", "Isotope", or "Seed"

    Returns:
        Tuple containing the Category, Isotope, and/or Seed values identified
        for the old label format.
    """

    old_label = label_val.strip()
    # Some files use 'background', others use 'Background'.
    if old_label.lower() == BACKGROUND_LABEL.lower():
        old_label = BACKGROUND_LABEL

    if label_level == "Category":
        return (old_label,)

    if label_level == "Isotope":
        category = _find_category(old_label)
        return (category, old_label)

    if label_level == "Seed":
        special_cases = SEED_TO_ISOTOPE_SPECIAL_CASES.keys()
        exact_match = old_label in special_cases
        first_partial_match = None
        if not exact_match:
            first_partial_match = next((x for x in special_cases if x in old_label), None)
        if exact_match:
            isotope = SEED_TO_ISOTOPE_SPECIAL_CASES[old_label]
        elif first_partial_match:
            isotope = SEED_TO_ISOTOPE_SPECIAL_CASES[first_partial_match]
        else:
            isotope = _find_isotope(old_label, verbose)
        category = _find_category(isotope)
        return (category, isotope, old_label)


def isotope_name_is_valid(isotope: str):
    """Validate whether the given string contains a properly formatted radioisotope name.

    Note that this function does NOT look up a string to determine if the string corresponds
    to a radioisotope that actually exists, it just checks the format.

    The regular expression used by this function looks for the following (in order):

    - 1 capital letter
    - 0 to 1 lowercase letters
    - 1 to 3 numbers
    - an optional "m" for metastable

    Examples of properly formatted isotope names:

    - Y88
    - Ba133
    - Ho166m

    Args:
        isotope: string containing the isotope name

    Returns:
        Bool representing whether the name string is valid
    """
    validator = re.compile(r"^[A-Z]{1}[a-z]{0,1}[0-9]{1,3}m?$")
    other_valid_names = ["fiestaware"]
    match = validator.match(isotope)
    is_valid = match is not None or \
        isotope.lower() in other_valid_names
    return is_valid
