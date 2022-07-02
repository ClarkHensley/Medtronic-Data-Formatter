#!/usr/bin/env python3

"""
This file will plot the data after it has been extracted
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv as csv_lib
import statistics as stats

from copy import deepcopy
from scipy.special import erfc

from SettingsGenerator import generateSettings, generateDatasetsAndGroups, generateRelativeConstants
from DataExtractor import extractFromAll
from FormattingClasses import StrikeSet, TestSet, DataSet

def main():

    source_directory = os.path.dirname(os.path.realpath(__file__))

    settings = generateSettings(source_directory)

    datasets_dict, groups_dict = generateDatasetsAndGroups(source_directory)

    plt.rcParams.update({"font.size": settings["TEXT-SIZE"]})
    dataset = settings["DATASET"]
    groups_list = datasets_dict[dataset]

    DELIMITER, raw_data_directory, results_directory, tests_directory = generateRelativeConstants(source_directory, dataset)

    groups_dirs = []
    needToGenerate = False
    for group in groups_list:
        outer_dir = os.path.join(raw_data_directory, group)
        for test in groups_dict[group]["tests"]:
            desired_dir = os.path.join(outer_dir, test)
            if not os.path.exists(desired_dir):
                needToGenerate = True
                break

            if not needToGenerate:
                strikes_list = list(filter(lambda c: c.endswith(".csv"), os.listdir(desired_dir)))
                strikes_list.sort()

                if not len(strikes_list) >= 1:
                    needToGenerate = True
                    break

            if needToGenerate:
                print(f"INFO: Some or All Data for Dataset {dataset} have not been extracted. Beginning extraction process")
                extractFromAll(settings, datasets_dict, groups_dict)
                break

    full_groups = {}

    max_num_tests = 0
    for group in groups_list:
        full_groups[group] = []
        tests_dict = groups_dict[group]
        tests = tests_dict["tests"]
        max_num_tests = max(max_num_tests, len(tests))
        for test in tests:
            full_groups[group].append(os.path.join(tests_directory, test))

    num_groups = len(groups_list)

    this_dataset = DataSet(num_groups, max_num_tests)

    # Now, go through each directory and format the data
    for g, group in enumerate(groups_list):
        raw_group_dir = os.path.join(raw_data_directory, group)
        group_dir = os.path.join(results_directory, group)
        if not os.path.exists(group_dir):
            os.mkdir(group_dir)
        this_dataset.data_record[group] = {}
        for t, test in enumerate(groups_dict[group]["tests"]):
            raw_test_dir = os.path.join(raw_group_dir, test)
            test_dir = os.path.join(group_dir, test)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            os.chdir(test_dir)
            strikes_list = list(filter(lambda c: c.endswith(".csv"), os.listdir(raw_test_dir)))
            strikes_list.sort()

            this_test = TestSet(test, len(strikes_list))
            accepted_strikes = {}
            for csv in strikes_list:
                short_csv = csv.split(".csv")[0]
                short_csv = short_csv.split("RAW-")[1]
                with open(os.path.join(raw_test_dir, f"{short_csv}-TIME-MULTIPLE.txt"), "r") as tmtxt:
                    time_multiple = int(tmtxt.read())

                with open(os.path.join(raw_test_dir, f"{short_csv}-INDEX.txt"), "r") as indtxt:
                    s = int(indtxt.read()) - 1

                data = pd.read_csv(os.path.join(raw_test_dir, csv), dtype=float).to_numpy(dtype=float)
                this_strike = StrikeSet(data, group, short_csv, settings, time_multiple)
                this_test.addStrikeSet(this_strike, s, settings)

                for i, _ in enumerate(data):
                    this_test.strike_set.time_arr[i] = this_test.strike_set.time_multiple * (float(data[i, 0]) - float(data[0, 0]))

                    this_test.strike_set.impact_arr[i] = float(data[i, 1]) * settings["kN"]
                    if this_test.strike_set.accelerometer_present:
                        this_test.strike_set.accel_arr[i] = float(data[i, 2]) / settings["mV_to_a"]

                this_test.characterizeWaveform(settings, s)
                # If there are major outliers, use iterative smoothing once to try to fixthose.
                if this_test.force_max_arr[s] >= settings["FORCE-UPPER-REASONABLE"]:
                    this_test.strike_set.smoothIteratively(groups_dict[group]["threshold"], groups_dict[group]["iterations"])
                    this_test.characterizeWaveform(settings, s)
                this_test.strike_set.smoothCurve(settings)
                this_test.characterizeWaveform(settings, s)

                # Check if the strike is rejected, otherwise display it
                if not (settings["FORCE-LOWER-REASONABLE"] <= this_test.force_max_arr[s] <= settings["FORCE-UPPER-REASONABLE"]) or not (this_test.area_arr[s] >= settings["AREA-LOWER-REASONABLE"]):
                    this_test.rejected_strikes.append(s)
                    print(f"REJECTED STRIKE {short_csv}: OUT OF BOUNDS")

                else:
                    this_test.updateData()
                    this_test.initialAppend()
                        # Plot per-strike
                    this_test.strike_set.plotAllData(settings)

            # Reject by Chauvenet's Criterion:
            # Do it for both Area and Max Force
            this_dataset.calculateStats(this_test, (g, t))

            # Plot per-test
            this_dataset.data_record[group][test] = {
                "area": deepcopy(this_test.area_arr),
                "force_max": deepcopy(this_test.force_max_arr),
                "init_slope": deepcopy(this_test.init_slope_arr),
                "wavelength": deepcopy(this_test.wavelength_arr),
                "rejected_strikes": deepcopy(this_test.rejected_strikes)
            }

            this_test.plotAllData(settings)

        #Now manage the test now that it's filled
        for t, test in enumerate(list(this_dataset.data_record[group].keys())):
            selected_area = this_dataset.data_record[group][test]["area"]
            selected_force_max = this_dataset.data_record[group][test]["force_max"]
            for key in list(selected_area.keys()):
                if key not in this_dataset.data_record[group][test]["rejected_strikes"]:
                    if ((1 / 2 * len(selected_area)) < erfc(np.abs(selected_area[key] - this_dataset.area_mean_arr[g, t]) / this_dataset.area_stdev_arr[g, t])) or ((1 / 2 * len(selected_force_max)) < erfc(np.abs(selected_force_max[key] - this_dataset.force_max_mean_arr[g, t]) / this_dataset.force_max_stdev_arr[g, t])):
                        this_dataset.data_record[group][test]["rejected_strikes"].append(key)
                        if key < 9:
                            print(f"REJECTED STRIKE {test}_0{key + 1}: CHAUVENET'S CRITERION")
                        else:
                            print(f"REJECTED STRIKE {test}_{key + 1}: CHAUVENET'S CRITERION")
                        this_dataset.delRow((g, t))

        # Plot per-group
        os.chdir(group_dir)
        this_dataset.plotAllRawData(group, settings)


    # Record per-Dataset
    os.chdir(results_directory)
    this_dataset.plotAllMeanData(settings)
    final_means_and_stdevs = []
    final_means_and_stdevs.append(("Implant", "Area Mean", "Force Mean", "Slope Mean", "Length Mean", "Area STDdev", "Force STDdev", "Slope STDdev", "Length STDdev"))
    for g, group in enumerate(groups_list):
        for t, test in enumerate(this_dataset.data_record[group]):
            final_means_and_stdevs.append((test, this_dataset.area_mean_arr[g][t], this_dataset.force_max_mean_arr[g][t], this_dataset.init_slope_mean_arr[g][t], this_dataset.wavelength_mean_arr[g][t], this_dataset.area_stdev_arr[g][t], this_dataset.force_max_stdev_arr[g][t], this_dataset.init_slope_stdev_arr[g][t], this_dataset.wavelength_stdev_arr[g][t]))
    with open(f"Final-Data-For-{dataset}.csv", "w", newline="") as newcsv:
        writer = csv_lib.writer(newcsv)
        writer.writerows(final_means_and_stdevs)

    print("FINISHED")

if __name__ == "__main__":
    main()
