#!/usr/bin/env Python3

"""
This file reads from settings.txt, formats the data,
and implements it into settings.json, in addition to some "universal constants"
"""
import os
import json

def formatData():

    # Absolute Path of this file
    directory = os.path.dirname(os.path.realpath(__file__))
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

            try:
                data = int(data)

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

