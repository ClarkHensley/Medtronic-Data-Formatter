#!/usr/bin/env python3

import os

"""
The functions in this file manage the serialization of text files, which allows for plain-text configuration of the Medtronic Project
"""

def generateDatasetsAndGroups(directory):

    DATASETS = {}
    GROUPS = {}

    # File Paths
    datasets_file = os.path.join(directory, "datasets.txt")
    groups_file = os.path.join(directory, "groups.txt")

    # Process Datasets
    with open(datasets_file, "r") as df:
        raw_datasets = df.readlines()

    # format each line
    for line in raw_datasets:
        if line == "" or line.isspace() or line.startswith("#"):
            continue

        else:
            line = line.strip()
            vals = line.split(":")

            # No extra settings, either a header line or a dataset definition
            dataset = vals[0]
            dataset = dataset.strip()
            if (dataset[0] == "'" or dataset[0] == '"') and (dataset[-1] == "'" or dataset[-1] == '"'):
                dataset = dataset[1:-1]
                dataset = dataset.strip()

            if len(vals) > 2:
                groups = ":".join(vals[1:])
            else:
                groups = vals[1]

            if groups == "":
                # Do nothing with header "Datasets:" line
                continue

            else:
                # Groups for the dataset
                groups = groups.split(",")
                DATASETS[dataset] = []

                for group in groups:
                    group = group.strip()
                    if (group[0] == "'" or group[0] == '"') and (group[-1] == "'" or group[-1] == '"'):
                        group = group[1:-1]
                        group = group.strip()

                    DATASETS[dataset].append(group)

    with open(groups_file, "r") as gf:
        raw_groups = gf.readlines()

    for line in raw_groups:
        if line == "" or line.isspace() or line.startswith("#"):
            continue

        else:
            line = line.strip()
            vals = line.split(":")

            group = vals[0]
            group = group.strip()
            if (group[0] == "'" or group[0] == '"') and (group[-1] == "'" or group[-1] == '"'):
                group = group[1:-1]
                group = group.strip()

            if len(vals) > 2:
                tests = ":".join(vals[1:])
            else:
                tests = vals[1]

            if tests == "":
                # Do nothing with header "Groups:" line
                continue

            else:
                # Split on the commas
                tests = tests.split(",")

                # First two values should alwasy be Threshold and Iterations. Find just the number parts
                threshold = tests[0].strip()
                iterations = tests[1].strip()
                tests = tests[2:]

                if not threshold.isnumeric():
                    threshold = threshold.split(": ")[1].strip()
                    if not threshold.isnumeric():
                        print("ERROR: invalid format of data in groups.txt. You must specify a Threshold value for each group")
                        raise SystemExit
                threshold = int(threshold)

                if not iterations.isnumeric():
                    iterations = iterations.split(": ")[1].strip()
                    if not iterations.isnumeric():
                        print("ERROR: invalid format of data in groups.txt. You must specify an Iteration value for each group")
                        raise SystemExit
                iterations = int(iterations)

                GROUPS[group] = {"threshold": threshold, "iterations": iterations, "tests": []}
                for test in tests:
                    test = test.strip()
                    if (test[0] == "'" or test[0] == '"') and (test[-1] == "'" or test[-1] == '"'):
                        test = test[1:-1]
                        test = test.strip()

                    GROUPS[group]["tests"].append(test)

    return DATASETS, GROUPS


def generateSettings(directory):

    # Absolute Path of this file
    settings_file = os.path.join(directory, "settings.txt")

    with open(settings_file, "r") as sf:
        settings = sf.readlines()

    # format each line
    settings_dict = {}
    inner_dicts = {}
    started_dict = False
    current_dict = ""
    for line in settings:
        if line == "" or line.isspace() or line.startswith("#"):
            continue

        elif "{" in line:
            started_dict = True
            current_dict = line.split(":")[0]
            inner_dicts[current_dict] = {}
        elif "}" in line:
            started_dict = False
            current_dict = ""
        else:
            if started_dict:
                vals = line.split(",")
                key = vals[0].strip()
                value = int(vals[1])
                inner_dicts[current_dict][key] = value

            else:

                vals = line.split(":")

                # If, somehow, there is an extra colon in the data in the setting, we'll rejoin that
                setting = vals[0]
                setting = setting.strip()
                if len(vals) > 2:
                    data = ":".join(vals[1:])
                else:
                    data = vals[1]

                data = data.strip()

                bool_data = {"true": True, "false": False}
                if data.lower() in bool_data.keys():

                    data = bool_data[data.lower()]

                else:

                    try:
                        data = float(data)

                    except ValueError:
                        pass

                settings_dict[setting] = data

    # Add any Dictionaries we created
    for key, value in list(inner_dicts.items()):
        settings_dict[key] = value
    # Now, we need to add some universal constants to the dictionary

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

    settings_dict["fig_size"] = (19.2, 10.8)

    return settings_dict

def generateRelativeConstants(source_directory, dataset):

    # "\" for Windows
    if os.name == "nt":
        DELIMITER = "\\"
    # "/" for Mac/Linux
    else:
        DELIMITER = "/"

    main_results_directory = os.path.join(source_directory, "Results")
    if not os.path.exists(main_results_directory):
        os.mkdir(main_results_directory)

    results_directory = os.path.join(main_results_directory, f"{dataset}")
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    main_raw_data_directory = os.path.join(source_directory, "RawData")
    if not os.path.exists(main_raw_data_directory):
        os.mkdir(main_raw_data_directory)

    raw_data_directory = os.path.join(main_raw_data_directory, f"{dataset}")
    if not os.path.exists(raw_data_directory):
        os.mkdir(raw_data_directory)

    tests_directory = os.path.join(source_directory, "Tests")

    return DELIMITER, raw_data_directory, results_directory, tests_directory

if __name__ == "__main__":
    print("SettingsGenerator.py is not meant to be run on its own.")

