"""
Updated Medtronic Data Formatter
Maintainer: Clark Hensley (ch3136)
Upstream: https://github.com/ClarkHensley/Medtronic-Data-Formatter
"""

import os
import sys
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erfc

from FormattingClasses import StrikeSet, TestSet, DataSet


###############################################################################
# Structure of the filesystem:
#   Runs include multiple Tests
#       Tests include multipe CSV files, multiple Strikes
###############################################################################


def main():

    # Directory of this file
    source_directory = os.path.dirname(os.path.realpath(__file__))

    # Firstly, generate the settings
    formatData(source_directory)

    # And read from the data json
    settings = dictFromJson("settings.json")

    plt.rcParams.update({'font.size': settings["TEXT-SIZE"]})

    DATASET = settings["DATASET"]

    # Data Directory of this file
    dataformatter_directory = os.path.join(source_directory, "Dataformatter")
    os.chdir(dataformatter_directory)

    if not os.path.exists(os.path.join(dataformatter_directory, "Data")):
        os.mkdir(os.path.join(dataformatter_directory, "Data"))

    if not os.path.exists(os.path.join(dataformatter_directory, f"Data/{DATASET}")):
        os.mkdir(os.path.join(dataformatter_directory, f"Data/{DATASET}"))

    # The list of "runs" which includes multiple tests each
    groups = dictFromJson("groups.json")

    # The list of those tests
    groups_list = groups[DATASET]

    # The strikes in each
    database = dictFromJson("database.json")

    # We'll keep a list of each dataset we're testing
    full_groups = {}

    # First, we're going to get all the data out of the CSVs for each test
    dim2 = 0
    for group in groups_list:
        full_groups[group] = []
        CSV_dict = dict(database[group])
        FOLDER = CSV_dict["FOLDER"]
        TESTS = list(CSV_dict["TESTS"])
        dim2 = max(dim2, len(TESTS))
        for test in TESTS:
            full_groups[group].append(os.path.join(FOLDER, test))

    dim1 = len(groups_list)

    total_dataset = DataSet(dim1, dim2)

    for g, group in enumerate(full_groups):

        # Set up each group, one test at a time

        print()
        print(f"BEGINNING GROUP {group}")
        print()

        for t, test in enumerate(full_groups[group]):

            csv_path = os.path.join(source_directory, test)

            test = test.split("/")[-1]

            csv_list = os.listdir(csv_path)
            csv_list = list(filter(lambda c: c.endswith(".csv"), csv_list))
            csv_list.sort()

            print()
            print(f"BEGINNING TEST {test}")
            print()

            total_dataset.strike_record[test] = {}
            total_dataset.strike_count[test] = 0

            test_size = len(csv_list)

            this_test = TestSet(test, test_size)

            for s, csv in enumerate(csv_list):

                print()
                print(f"BEGINNING STRIKE {s + 1}")

                this_strike_set = formatCSV(csv_path, csv, group, settings)
                
                this_test.addStrikeSet(this_strike_set, s, settings)

                for ind in range(len(this_test.strike_set.data) - 250):

                    if this_test.strike_set.strike_triggered:
                        break

                    current_slope = this_test.getCurrentSlope(ind)

                    # This finds if the strike has occured
                    if (np.all(this_test[ind - 250:ind + 250, 1] != "5") and float(this_test[ind, 1]) >= float(settings["FORCE-LOWER-REASONABLE"]) / float(settings["kN"]) and current_slope >= float(settings["SLOPE-LOWER-LIMIT"])):

                        this_test.strike_set.strike_triggered = True
                        subset_arr = this_test.strike_set[(ind - int(this_test.strike_set.inc) + int(this_test.shift)): (ind + int(this_test.strike_set.inc) + int(this_test.shift))]

                        for i, _ in enumerate(subset_arr):

                            this_test.strike_set.time_arr[i] = this_test.strike_set.time_multiple * (float(this_test[i, 0]) - float(this_test[0, 0]))

                            #print("Before:", float(this_test[i, 1]), ",", float(settings["kN"]))
                            this_test.strike_set.impact_arr[i] = float(this_test[i, 1]) * float(settings["kN"]) * 100
                            #print("After:", this_test.strike_set.impact_arr[i], "\n")

                            if this_test.strike_set.accelerometer_present:
                                this_test.strike_set.accel_arr[i] = float(this_test[i, 2]) / float(settings["mV_to_a"])

                        if this_test.strike_set.fitting_arr[ind]:
                            this_test.strike_set.fitCurve(settings)

                        this_test.characterizeWaveform(settings, s)

                        this_test.updateData()

                if not this_test.strike_set.strike_triggered:
                    print(f"WARNING: STRIKE {s + 1} WAS REJECTED")
                    print()
                    this_test.rejected_strikes.append(s)

                else:
                    this_test.strike_count += 1
                    this_test.initialAppend()

                print(f"ENDING STRIKE {s + 1}\n")
                print()

            # Ensure the data are as expected, reject and remove outstanding values
            #  for f in range(int(total_dataset.strike_count[(g, t)])):
            #
            #      if not(settings["FORCE-LOWER-REASONABLE"] >= this_test.force_max_arr[f] >= settings["FORCE-LOWER-REASONABLE"] and this_test.area_arr[f] >= settings["AREA-LOWER-LIMIT"]):
            #
            #          this_test.strike_set.rejection_arr[f] = True
            #          print(f"REJECTED STRIKE {f + 1}, OUT OF BOUNDS")

            #  # Possibly not the best way to filter a numpy array, but we already store the rejection array, so this is a good use for it
            #  this_test.area_arr = this_test.area_arr[this_test.strike_set.rejection_arr]
            #  this_test.force_max_arr = this_test.force_max_arr[this_test.strike_set.rejection_arr]
            #  this_test.init_slope_arr = this_test.init_slope_arr[this_test.strike_set.rejection_arr]
            #  this_test.wavelength_arr = this_test.wavelength_arr[this_test.strike_set.rejection_arr]

            this_test.finalize()

            total_dataset.calculateStats(this_test, (g, t))

            this_test.plotAllData(settings)

            del this_test



def formatData(directory):

    # Absolute Path of this file
    settings_file = os.path.join(directory, "settings.txt")

    with open(settings_file, "r") as sf:
        settings = sf.readlines()

    # format each line
    settings_dict = {}
    for setting in settings:
        if setting == "" or setting.isspace() or setting.startswith("#"):
            pass

        else:
            setting_vals = setting.split(":")

            # If, somehow, there is an extra colon in the data in the setting, we'll rejoin that
            setting = setting_vals[0]
            setting = setting.strip()
            if len(setting_vals) > 2:
                data = setting_vals[1:].join(":")
            else:
                data = setting_vals[1]

            data = data.strip()
            
            if data == "True" or data == "False":

                data = bool(data)

            else:

                try:
                    data = float(data)

                except ValueError:
                    pass

            settings_dict[setting] = data

    # Now, we need to add some universal constants to the json

    # According to the original file, this converts "voltage to lbf to kN" and "1V = 100lbf for our sensor" whatever that means
    kN = 4.44822162

    # "converts milli-volts to acceleration"
    mV_to_a = 1.090

    # "picoscope range for the acceleration"
    max_av = 10

    # Sampling time rate (microseconds)
    timestep = 300

    # Residue for force
    residue = 1000

    settings_dict["kN"] = kN
    settings_dict["mV_to_a"] = mV_to_a
    settings_dict["max_av"] = max_av
    settings_dict["timestep"] = timestep
    settings_dict["residue"] = residue

    # Dump to json
    with open("settings.json", "w") as sj:
        json.dump(settings_dict, sj)


def dictFromJson(file):
    """ Attempt to open a .json file, which will be converted into a python Dictionary."""

    # Attempt to create a dictionary from the file and return it
    try:
        with open(file, "r") as h:
            h_content = h.read()
            temp = json.loads(h_content)

        return temp

    except FileNotFoundError:
        sys.exit(f"{file} JSON file could not be found.")


def formatCSV(path, csv, group, settings):
    """
    Remove invalid values from the CSV, take care of under- or over-flow errors
    """

    csv_full_path = os.path.join(path, csv)

    data = pd.read_csv(csv_full_path, delimiter=",", dtype=str).to_numpy(dtype=str)

    print(data.shape)
    print(len(data))
    if(len(data) >= 40000):
        step = int(len(data) / 20000)
        print(step)
        data = data[::step, :]
    print(len(data))
    sys.exit()

    fitting_arr = np.array([False for _ in range(len(data))])

    time_storage = data[0, 0]

    data = data[3:]

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


if __name__ == "__main__":
    main()

