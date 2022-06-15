#!/usr/bin/env python3
"""
Updated Medtronic Data Formatter
Maintainer: Clark Hensley (ch3136)
Upstream: https://github.com/ClarkHensley/Medtronic-Data-Formatter
"""

import os
import sys
import random

import numpy as np
import pandas as pd
import csv as csv_lib
import matplotlib.pyplot as plt

from copy import deepcopy

from FormattingClasses import StrikeSet, TestSet
from SettingsGenerator import generateSettings, generateDatasetsAndGroups, generateRelativeConstants

###############################################################################
# Structure of the filesystem:
#   Datasets include multiple Groups
#       Groups include multiple Tests
#           Tests include multiple Strikes (That is, multiple CSV files)
###############################################################################


def extractFromAll(settings=None, datasets_dict=None, groups_dict=None):

    # Directory of this file
    source_directory = os.path.dirname(os.path.realpath(__file__))

    # Extract settings from text file
    if settings is None:
        settings = generateSettings(source_directory)

    if datasets_dict is None or groups_dict is None:
        datasets_dict, groups_dict = generateDatasetsAndGroups(source_directory)

    dataset = settings["DATASET"]
    groups_list = datasets_dict[dataset]

    DELIMITER, raw_data_directory, results_directory, tests_directory = generateRelativeConstants(source_directory, dataset)

    # We'll keep a list of each dataset we're testing
    full_groups = {}

    # First, we're going to get all the data out of the CSVs for each test
    max_num_tests = 0
    for group in groups_list:
        full_groups[group] = []
        tests_dict = groups_dict[group]
        tests = tests_dict["tests"]
        max_num_tests = max(max_num_tests, len(tests))
        for test in tests:
            full_groups[group].append(os.path.join(tests_directory, test))

    num_groups = len(groups_list)

    # After getting the data from the Datasets directory, change directory back to <SOURCE>/Results/{DATASET} to easily save the results

    print("START")

    for g, group in enumerate(full_groups):
        # Set up each group, one test at a time
        extractFromGroup(group, g, full_groups, source_directory, DELIMITER, raw_data_directory, settings)

    print("DONE")
    return

def extractFromGroup(group, g, full_groups, source_directory, DELIMITER, raw_data_directory, settings):
    # Extrac data From each group, test by test, plot and store data afterwards
    print()
    print(f"\tBEGINNING GROUP {group}")

    current_group_dir = os.path.join(raw_data_directory, f"{group}")
    if not os.path.exists(current_group_dir):
        os.mkdir(current_group_dir)

    for t, test in enumerate(full_groups[group]):
        extractFromTest(source_directory, group, g, test, t, DELIMITER, current_group_dir, settings)

def extractFromTest(source_directory, group, g, test, t, DELIMITER, current_group_dir, settings):
    # Extract data from each test, gather average data across all strikes per test
    csv_path = os.path.join(source_directory, test)

    test = test.split(DELIMITER)[-1]

    csv_list = list(filter(lambda c: c.endswith(".csv"), os.listdir(csv_path)))
    csv_list.sort()

    print()
    print(f"\t\tBEGINNING TEST {test}")
    print()

    current_test_dir = os.path.join(current_group_dir, f"{test}")
    if not os.path.exists(current_test_dir):
        os.mkdir(current_test_dir)
    os.chdir(current_test_dir)

    test_size = len(csv_list)

    this_test = TestSet(test, test_size)

    for s, csv in enumerate(csv_list):
        this_test = extractFromStrike(this_test, s, csv, csv_path, group, settings)

    del this_test

def extractFromStrike(this_test, s, csv, csv_path, group, settings):
    # Go through each CSV, find and extract data from each strike, or warn about a missing strike
    print(f"\t\t\tBEGINNING STRIKE {s + 1}")

    this_strike_set = extractFromCSV(csv_path, csv, group, settings)

    this_test.addStrikeSet(this_strike_set, s, settings)

    for ind in range(len(this_test.strike_set.data) - 250):

        if this_test.strike_set.strike_triggered:
            break

        current_slope = this_test.getCurrentSlope(ind)

        # This finds if the strike has occured
        if np.all(this_test[ind - 250:ind + 250, 1] != "5") and float(this_test[ind, 1]) >= settings["FORCE-LOWER-REASONABLE"] / settings["kN"] and current_slope >= settings["SLOPE-LOWER-LIMIT"]:

            subset_arr = this_test.strike_set[(ind - int(this_test.strike_set.inc) + int(this_test.shift)): (ind + int(this_test.strike_set.inc) + int(this_test.shift))]
            with open(f"RAW-{csv}", "w", newline="") as newcsv:
                writer = csv_lib.writer(newcsv)
                writer.writerows(subset_arr)

            short_csv = csv.split(".csv")[0]
            with open(f"{short_csv}-TIME-MULTIPLE.txt", "w") as tmtxt:
                tmtxt.write(str(this_test.strike_set.time_multiple))

            with open(f"{short_csv}-INDEX.txt", "w") as indtxt:
                indtxt.write(str(s + 1))

            return this_test


    if not this_test.strike_set.strike_triggered:
        print()
        print(f"\t\t\t\tWARNING: STRIKE {s + 1} WAS REJECTED: Strike Not Triggered")
        print()
        return this_test

def extractFromCSV(path, csv, group, settings):
    """
    Remove invalid values from the CSV, take care of under- or over-flow errors
    """

    csv_full_path = os.path.join(path, csv)

    data = pd.read_csv(csv_full_path, delimiter=",", dtype=str).to_numpy(dtype=str)

    if(len(data) >= 40000):
        step = int(len(data) / 20000)
        data = data[::step, :]

    fitting_arr = np.array([False for _ in range(len(data))])

    time_storage = str(data[0, 0])

    # Take out the first row of data (the column headers)
    data = data[1:, :]

    # here we format the data
    for d, datum in enumerate(data):
        if datum[1] == "âˆž" or datum[1] == "∞":
            data[d, 1] = "5"
            fitting_arr[d] = True

        if datum[1] == "-∞" or float(datum[1]) < 0:
            data[d, 1] = "0"
            fitting_arr[d] = True

        if len(datum) == 3:
            if datum[2] == "âˆž" or datum[2] == "∞":
                data[d, 2] = str(settings["max_av"])

            if datum[2] == "-∞":
                data[d, 2] = str(-1 * settings["max_av"])

    return StrikeSet(data, group, csv, settings, time_storage)

if __name__ == "__main__":
    extractFromAll()

