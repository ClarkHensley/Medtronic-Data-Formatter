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
import matplotlib.colors as mcolors
import seaborn as sns

from copy import deepcopy
from FormattingClasses import StrikeSet, TestSet, DataSet
from SettingsGenerator import generateSettings, generateDatasetsAndGroups

###############################################################################
# Structure of the filesystem:
#   Datasets include multiple Groups
#       Groups include multiple Tests
#           Tests include multiple Strikes (That is, multiple CSV files)
###############################################################################


def extractFromAll(settings=None, datasets_dict=None, groups_dict=None):

    # "\" for Windows
    if os.name == "nt":
        DELIMITER = "\\"
    # "/" for Mac/Linux
    else:
        DELIMITER = "/"

    # Directory of this file
    source_directory = os.path.dirname(os.path.realpath(__file__))

    # Extract settings from text file
    if settings is None:
        settings = generateSettings(source_directory)

    if datasets_dict is None or groups_dict is None:
        datasets_dict, groups_dict = generateDatasetsAndGroups(source_directory)

    plt.rcParams.update({'font.size': settings["TEXT-SIZE"]})

    dataset = settings["DATASET"]
    groups_list = datasets_dict[dataset]

    tests_directory = os.path.join(source_directory, "Tests")

    main_results_folder = os.path.join(source_directory, "Results")
    if not os.path.exists(main_results_folder):
        os.mkdir(main_results_folder)

    results_dir = os.path.join(main_results_folder, f"{dataset}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)


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

    total_dataset = DataSet(num_groups, max_num_tests)

    # After getting the data from the Datasets directory, change directory back to <SOURCE>/Results/{DATASET} to easily save the results

    print("START")

    for g, group in enumerate(full_groups):
        # Set up each group, one test at a time
        total_dataset = extractFromGroup(total_dataset, group, g, full_groups, source_directory, DELIMITER, results_dir, settings)

    print("DONE")
    raise SystemExit

    ###############################################3
    group_list = list(total_dataset.data_record.keys())
    xlabel = "Group"

    mean_values = [
        ("area", "Mean Area Under the Impulse Curve", "Area (kN * us)"),
        ("force_max", "Mean Peak Force", "Force (kN)"),
        ("init_slope", "Mean Initial Slope of the Wave", "Slope (kN / us)"),
        ("wavelength", "Mean Duration of the Impact Event", "Duration (us)")
        ]

    for value in mean_values:
        (key, title, ylabel) = value

        if key == "area":
            init_data = deepcopy(total_dataset.area_mean_arr)
        elif key == "force_max":
            init_data = deepcopy(total_dataset.force_max_mean_arr)
        elif key == "init_slope":
            init_data = deepcopy(total_dataset.init_slope_mean_arr)
        elif key == "wavelength":
            init_data = deepcopy(total_dataset.wavelength_mean_arr)

        max_len = 0
        for data in init_data:
            max_len = max(max_len, len(data))

        data_dict = {}

        for i, datum in enumerate(init_data):
            temp = np.empty((max_len))
            for j, d in enumerate(datum):
                temp[j] = d
            data_dict[group_list[i]] = temp

        data = pd.DataFrame(data=data_dict)

        plotDataSet(title, xlabel, ylabel, data, results_dir, settings)

    #final_csv_columns = ["Implant", "Mean Area", "Mean Force", "Mean Slope", "Mean Length", "StDev Area", "StDev Force", "StDev Slope", "StDev Length"]
    #final_record = np.array((len(final_csv_columns)))

    #for i, g_name in enumerate(groups_list):
        #final_record = np.vstack([final_record, [g_name, total_dataset.area_mean_arr[i], total_dataset.force_max_mean_arr[i], total_dataset.init_slope_mean_arr[i], total_dataset.wavelength_mean_arr[i], total_dataset.area_stdev_arr[i], total_dataset.force_max_stdev_arr[i], total_dataset.init_slope_stdev_arr[i], total_dataset.wavelength_stdev_arr[i]]])

    #final_df = pd.DataFrame(final_record, columns=final_csv_columns)
    #final_df.to_csv(os.path.join(results_dir, f"{DATASET}_record_data.csv"), encoding="utf-8")

    print("DONE")
    #########################################

def extractFromGroup(total_dataset, group, g, full_groups, source_directory, DELIMITER, results_dir, settings):
    # Extrac data From each group, test by test, plot and store data afterwards
    print()
    print(f"\tBEGINNING GROUP {group}")

    current_group_dir = os.path.join(results_dir, f"{group}")
    if not os.path.exists(current_group_dir):
        os.mkdir(current_group_dir)

    total_dataset.data_record[group] = {}

    for t, test in enumerate(full_groups[group]):
        total_dataset = extractFromTest(source_directory, group, g, test, t, total_dataset, DELIMITER, current_group_dir, settings)

    return total_dataset

    ##################
    test_list = []

    for name in total_dataset.data_record[group]:
        new_name = " ".join(name.split(" ")[1:])
        test_list.append(new_name)

    xlabel = "Test"

    values = [
            ("area", f"Area Under the Impulse Curve for {group}", "Area (kN * us)"),
            ("force_max", f"Peak Force for {group}", "Force (kN)"),
            ("init_slope", f"Initial Slope of the Wave for {group}", "Slope (kN / us)"),
            ("wavelength", f"Duration of the Impact Event for {group}", "Duration (us)")
            ]

    for value in values:
        (key, title, ylabel) = value

        max_len = 0
        for data in total_dataset.data_record[group].values():
            max_len = max(max_len, len(data[key]))

        data_dict = {}

        for i, datum in enumerate(total_dataset.data_record[group].values()):
            temp = np.empty((max_len))
            for j, d in enumerate(datum[key]):
                temp[j] = d
            data_dict[test_list[i]] = temp

        data = pd.DataFrame(data=data_dict)

        plotDataSet(title, xlabel, ylabel, data, results_dir, settings)

    print(f"ENDING GROUP {group}")
    print()

    return total_dataset
    #########################

def extractFromTest(source_directory, group, g, test, t, total_dataset, DELIMITER, current_group_dir, settings):
    # Extract data from each test, gather average data across all strikes per test
    csv_path = os.path.join(source_directory, test)

    test = test.split(DELIMITER)[-1]
    total_dataset.data_record[group][test] = {}

    csv_list = os.listdir(csv_path)
    csv_list = list(filter(lambda c: c.endswith(".csv"), csv_list))
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
    return total_dataset

    ##################
    this_test.finalize(settings)

    total_dataset.calculateStats(this_test, (g, t))

    total_dataset.data_record[group][test]["area"] = deepcopy(this_test.area_arr)
    total_dataset.data_record[group][test]["force_max"] = deepcopy(this_test.force_max_arr)
    total_dataset.data_record[group][test]["init_slope"] = deepcopy(this_test.init_slope_arr)
    total_dataset.data_record[group][test]["wavelength"] = deepcopy(this_test.wavelength_arr)

    this_test.plotAllData(settings)

    del this_test

    return total_dataset
    ##################

def extractFromStrike(this_test, s, csv, csv_path, group, settings):
    # Go through each CSV, find and extract data from each strike, or warn about a missing strike
    print(f"\t\t\tBEGINNING STRIKE {s + 1}")

    this_strike_set = formatCSV(csv_path, csv, group, settings)

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
            #this_test = processStrikeSegment(s, ind, this_test, settings + int(this_test.strike_set.inc) + int(this_test.shift))
            return this_test


    if not this_test.strike_set.strike_triggered:
        print()
        print(f"\t\t\t\tWARNING: STRIKE {s + 1} WAS REJECTED: Strike Not Triggered")
        print()
        return this_test

    ##################
        this_test.rejected_strikes.append(s)

    else:
        this_test.strike_count += 1
        this_test.initialAppend()

    print(f"ENDING STRIKE {s + 1}\n")
    print()

    return this_test
    ##################

def processStrikeSegment(s, ind, this_test, settings):
    # Gather the data from the part of the CSV where the strike occurs, clean up the data, store it in a separate CSV, and return
    this_test.strike_set.strike_triggered = True
    subset_arr = this_test.strike_set[(ind - int(this_test.strike_set.inc) + int(this_test.shift)): (ind + int(this_test.strike_set.inc) + int(this_test.shift))]
    for i, _ in enumerate(subset_arr):

        this_test.strike_set.time_arr[i] = this_test.strike_set.time_multiple * (float(subset_arr[i, 0]) - float(subset_arr[0, 0]))

        this_test.strike_set.impact_arr[i] = float(subset_arr[i, 1]) * settings["kN"]

        if this_test.strike_set.accelerometer_present:
            this_test.strike_set.accel_arr[i] = float(subset_arr[i, 2]) / settings["mV_to_a"]

    if this_test.strike_set.fitting_arr[ind]:
        this_test.strike_set.fitCurve(settings)

    this_test.characterizeWaveform(settings, s)

    this_test.strike_set.smootheCurve(settings)

    this_test.updateData()

    return this_test

def formatCSV(path, csv, group, settings):
    """
    Remove invalid values from the CSV, take care of under- or over-flow errors
    """

    csv_full_path = os.path.join(path, csv)

    data = pd.read_csv(csv_full_path, delimiter=",", dtype=str).to_numpy(dtype=str)

    if(len(data) >= 40000):
        step = int(len(data) / 20000)
        data = data[::step, :]

    fitting_arr = np.array([False for _ in range(len(data))])

    time_storage = data[0, 0]

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

    return StrikeSet(data, fitting_arr, group, time_storage, csv, settings)

def plotDataSet(title, xlabel, ylabel, new_data, directory, settings):

    plt.figure(title, figsize=settings["fig_size"])
    plt.grid(True)
    plt.title(title)

    plt.rc("axes", titlesize=settings["TITLE-SIZE"])
    plt.rc("axes", labelsize=settings["LABEL-SIZE"])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    print(new_data)
    sns.violinplot(data=new_data)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, title) + ".png")
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    extractFromAll()

