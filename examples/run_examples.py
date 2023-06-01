# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Runs all Python files in the subfolders and reports whether they are successful."""
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

DIRS_TO_CHECK = ["data", "modeling", "visualization"]
FILENAME_KEY = "File"
RESULT_KEY = "Result"
SUCCESS_STR = "Success"
FAILURE_STR = "Fail"

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
    return_code = 0
    output = None
    try:
        output = subprocess.check_output(f"python {f} hide",
                                         stderr=subprocess.STDOUT,
                                         shell=True)

    except subprocess.CalledProcessError as e:
        if (bytes('Error', 'utf-8')) in e.output:
            print(e.output.decode())
            return_code = e.returncode

    results[i] = {
        FILENAME_KEY: os.path.relpath(f, example_dir),
        RESULT_KEY: SUCCESS_STR if not return_code else FAILURE_STR
    }
os.chdir(example_dir)

df = pd.DataFrame.from_dict(results, orient="index")
tabulated_df = tabulate(df, headers="keys", tablefmt="psql")
print(tabulated_df)

all_succeeded = all([x[RESULT_KEY] == SUCCESS_STR for x in results.values()])
sys.exit(0 if all_succeeded else 1)
