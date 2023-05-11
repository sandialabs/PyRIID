# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Runs all Python files in the subfolders and reports whether they are successful."""
import os
import subprocess
import sys
from pathlib import Path
from subprocess import DEVNULL

import pandas as pd
from tabulate import tabulate

DIRS_TO_CHECK = ["data", "modeling", "visualization"]
FILENAME_KEY = "File"
RESULT_KEY = "Result"
SUCCESS_STR = "Success"
FAILURE_STR = "Fail"

original_wdir = os.getcwd()
example_dir = Path(__file__).parent
os.chdir(example_dir)

files_to_run = []
for d in DIRS_TO_CHECK:
    file_paths = list(Path(d).rglob("*.py"))
    files_to_run.extend(file_paths)
files_to_run = sorted(files_to_run)

results = {}
n_tests = len(files_to_run)
for i, f in enumerate(files_to_run, start=1):
    print(f"Running example {i}/{n_tests}")
    proc = subprocess.Popen(f"python {f} hide",
                            stderr=DEVNULL,
                            stdout=DEVNULL,
                            shell=True)
    _, _ = proc.communicate()
    results[i] = {
        FILENAME_KEY: f,
        RESULT_KEY: SUCCESS_STR if not proc.returncode else FAILURE_STR
    }
os.chdir(original_wdir)

df = pd.DataFrame.from_dict(results, orient="index")
tabulated_df = tabulate(df, headers="keys", tablefmt="psql")
print(tabulated_df)

all_succeeded = all([x[RESULT_KEY] == SUCCESS_STR for x in results.values()])
sys.exit(0 if all_succeeded else 1)
