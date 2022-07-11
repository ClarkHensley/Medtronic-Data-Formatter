#!/usr/bin/env python3

import os

from copy import deepcopy
from scipy.special import erfc

import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
Classes for formatting Medtronic Data
"""

class StrikeSet:
    """
    This class just holds data for each CSV in one place
    """

    def __init__(self, data, group, strike, settings, time_storage):

        self.group = group
        self.name = strike

        if type(time_storage) == str:
            try:
                self.time_multiple = settings["TIME-DICT"][time_storage]

            except KeyError:
                print(f"ERROR! Unknown Time Units found: {time_storage}\nPlease add this with the correct value to the TIME-DICT setting in the settings.txt file.")
                raise SystemExit
        else:
            self.time_multiple = time_storage

        self.data = data
        self.accelerometer_present = len(data[0]) == 3

        self.timedelta = (float(data[1, 0]) - float(data[0, 0])) * self.time_multiple

        self.inc = settings["timestep"] / (2 * self.timedelta)

        self.arrsize = 2 * int(self.inc) + 2

        self.ratio = self.inc / 12000

        self.time_arr = np.zeros(shape=(self.arrsize))
        self.impact_arr = np.zeros(shape=(self.arrsize))
        self.accel_arr = np.zeros(shape=(self.arrsize))

        self.strike_triggered = False

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def fitCurve(self, settings):

        start_interpolation = False
        start_index = 0
        end_index = 0

        check_val = settings["FORCE-LOWER-BOUND"] * settings["kN"]

        for ind in range(2 * self.inc - 1):
            previous_val = self.impact_arr[ind]
            next_val = self.impact_arr[ind + 1]

            if start_interpolation:
                if 0 <= previous_val < settings["kN"] * 4 and abs(next_val - previous_val) / delta <= 2:
                    start_interpolation = False
                    end_index = ind
                    interpolation_slope = (self.impact_arr[end_index + 1] - self.impact_arr[start_index]) / (self.time_arr[end_index] - self.time_arr[start_index])

                    for new_ind in range(start_index, end_index):
                        self.impact_arr[new_ind] = self.impact_arr[new_ind - 1] * interpolation_slope * delta
                        ind = new_ind

                if ind == (2 * self.inc + 2):
                    start_interpolation = False
                    end_index = ind + 1
                    interpolation_slope = (0 - self.impact_arr[start_index]) / (self.time_arr[end_index] - self.time_arr[start_index])

                    for new_ind in range(start_index, end_index + 1):
                        self.impact_arr[new_ind] = self.impact_arr[new_ind - 1] + interpolation_slope * delta
                        ind = new_ind

            else:
                if abs(next_val - previous_val) / self.timedelta >= 10:
                    start_interpolation = True
                    start_index = ind - settings["residue"]

            if self.impact_arr[self.arrsize - 3] < check_val:
                self.impact_arr[self.arrsize - 3] = check_val

    def smoothCurve(self, settings):

        slopes = np.zeros(shape=(self.arrsize - 1))
        for i in range(0, self.arrsize - 1):
            slopes[i] = (self.impact_arr[i] - self.impact_arr[i - 1]) / self.timedelta

        slopes_mean = stats.mean(slopes)
        slopes_stdev = stats.stdev(slopes)
        mask = [True for _ in range(len(self.impact_arr))]

        for i in range(len(slopes)):
            if 1 / (2 * len(slopes)) > erfc(abs(slopes[i] - slopes_mean) / slopes_stdev):
                mask[i + 1] = False
        self.fitToMask(mask)

        # Final mask to clear out extra ending zeros
        last_mask = [(self.time_arr[i] != 0) for i in range(1, len(self.time_arr))]
        last_mask.insert(0, True)
        self.fitToMask(last_mask)

    def smoothIteratively(self, threshold, iterations):

        for _ in range(iterations):
            old_size = len(self.impact_arr)
            slopes = np.zeros(shape=(old_size))
            for i in range(1, old_size):
                slopes[i] = (self.impact_arr[i] - self.impact_arr[i - 1]) / self.timedelta

            mask = [True for _ in range(old_size)]

            for i in range(old_size):
                if abs(slopes[i]) > threshold or (slopes[i] == 0 and self.impact_arr[i] != 0):
                   mask[i] = False

            self.fitToMask(mask)
            new_arrsize = len(self.impact_arr)
            if new_arrsize == old_size:
                break

    def fitToMask(self, mask):
        prev_i = -1
        for i in range(len(mask)):
            if mask[i] is False:
                if prev_i == -1:
                    prev_i = i
            elif prev_i != -1:
                for j in range(prev_i, i):
                    mask[j] = False
                prev_i = -1

        self.impact_arr = self.impact_arr[mask]
        self.time_arr = self.time_arr[mask]

        if self.accelerometer_present:
            self.accel_arr = self.accel_arr[mask]
        self.arrsize = len(self.impact_arr)

class TestSet:
    """
    This class holds the data for a test
    """

    def __init__(self, name, dim):
        self.name = name

        self.size = dim

        self.strike_sets = {}

        self.rejected_strikes = []

        self.strike_count = 0

        self.shift = 0

        self.current_ind = -1

        self.strike_set = None

        self.inds = []

        # Total Data Values
        self.total_time_arr = None
        self.total_impact_arr = None
        self.total_accel_arr = None

        # Calculated Values
        self.area_arr = {}
        self.force_max_arr = {}
        self.init_slope_arr = {}
        self.wavelength_arr = {}

    def __getitem__(self, key):
        return self.strike_set[key]

    def __setitem__(self, key, value):
        self.strike_set[key] = value

    def addStrikeSet(self, strike_set, ind, settings):

        if self.strike_set is not None:
            del self.strike_set
            self.strike_set = None

        self.current_ind = ind

        self.inds.append(ind)

        self.strike_set = strike_set

        self.shift = int(settings["TIME-SHIFT"] * (2 * self.strike_set.inc / settings["timestep"]))

        self.strike_set.strike_triggered = False

    def updateData(self):
        if self.total_time_arr is None:
            self.total_time_arr = self.strike_set.time_arr
        else:
            self.total_time_arr = np.concatenate([self.total_time_arr, self.strike_set.time_arr])

        if self.total_impact_arr is None:
            self.total_impact_arr = self.strike_set.impact_arr
        else:
            self.total_impact_arr = np.concatenate([self.total_impact_arr, self.strike_set.impact_arr])

        if self.strike_set.accelerometer_present:
            if self.total_accel_arr is None:
                self.total_accel_arr = self.strike_set.accel_arr
            else:
                self.total_accel_arr = np.concatenate([self.total_accel_arr, self.strike_set.accel_arr])

    def initialAppend(self):
        self.strike_sets[self.current_ind] = deepcopy(self.strike_set)

    def getCurrentSlope(self, ind):

        return (float(self.strike_set.data[ind, 1]) - float(self.strike_set.data[ind - 1, 1])) / (float(self.strike_set.data[ind, 0]) - float(self.strike_set.data[ind - 1, 0]))

    def characterizeWaveform(self, settings, s):

        self.area_arr[s] = 0
        self.force_max_arr[s] = 0
        self.init_slope_arr[s] = 0
        self.wavelength_arr[s] = 0

        init_slope_points_cap = int(500 * self.strike_set.ratio)
        wave_duration = False

        for ind in range(self.strike_set.arrsize):
            self.area_arr[s] = self.area_arr[s] + (self.strike_set.impact_arr[ind] * self.strike_set.timedelta)

            if self.force_max_arr[s] < self.strike_set.impact_arr[ind]:
                self.force_max_arr[s] = self.strike_set.impact_arr[ind]

            if (self.strike_set.inc - (init_slope_points_cap / 2)) < ind + self.shift <= (self.strike_set.inc + (init_slope_points_cap / 2) - 1):

                self.init_slope_arr[s] += ((self.strike_set.impact_arr[ind] - self.strike_set.impact_arr[ind - 1]) / (self.strike_set.time_arr[ind] - self.strike_set.time_arr[ind - 1])) / init_slope_points_cap

        wave_start = 0
        for ind in range(self.strike_set.arrsize):
            if wave_duration:
                if self.strike_set.impact_arr[ind] <= settings["WAVE-END"]:
                    self.wavelength_arr[s] = (ind - wave_start) * (settings["timestep"] / (2 * self.strike_set.inc))
                    wave_duration = False

            else:
                if self.strike_set.impact_arr[ind] >= 3:
                    wave_duration = True
                    wave_start = ind

    def plotAllData(self, settings):

        # Plot Force v. Time
        time_v_force_data = {
                "title": f"Force v. Time for {self.name}",
                "xlabel": "Time (us)",
                "ylabel": "Force (kN)",
                "xdata": [],
                "ydata": [],
                "label": "Strike",
                "legend": settings["LEGEND"]
                }

        # Plot Accel v. Time
        if self.strike_set.accelerometer_present:
            time_v_accel_data = {
                "title": f"Accel v. Time for {self.name}",
                "xlabel": "Time (us)",
                "ylabel": "Acceleration (m/s^2)",
                "xdata": [],
                "ydata": [],
                "label": "Strike",
                "legend": settings["LEGEND"]
                }

        force_data = self.force_max_arr
        for val in self.rejected_strikes:
            force_data.pop(val)

        # Plot Force v. Strike
        strike_v_max_force_data = {
                "title": f"Max Force v. Strike Number for {self.name}",
                "xlabel": "Strike Number",
                "ylabel": "Force (kN)",
                "xdata": list(force_data.keys()),
                "ydata": list(force_data.values()),
                "label": "",
                "legend": False
                }

        for s, strike in enumerate(list(self.strike_sets.values())):

            time_v_force_data["xdata"].append(strike.time_arr)
            time_v_force_data["ydata"].append(strike.impact_arr)

            if strike.accelerometer_present:
                time_v_accel_data["xdata"].append(strike.time_arr)
                time_v_accel_data["ydata"].append(strike.accel_arr)

        self.plotData(time_v_force_data, settings)

        self.plotData(strike_v_max_force_data, settings)

        if self.strike_set.accelerometer_present:
            self.plotData(time_v_accel_data, settings)

    def plotData(self, data, settings):

        plt.figure(data["title"], figsize=settings["fig_size"])
        plt.grid(True)
        plt.title(data["title"])
        plt.xlabel(data["xlabel"])
        plt.ylabel(data["ylabel"])

        plt.rc("axes", titlesize=settings["TITLE-SIZE"])
        plt.rc("axes", labelsize=settings["LABEL-SIZE"])

        if type(data["xdata"][0]) != np.ndarray and type(data["ydata"][0]) != np.ndarray:
            plt.plot(data["xdata"], data["ydata"], marker=".", label=data["label"] + " 1")

        else:
            for i, _ in enumerate(data["xdata"]):
                plt.plot(data["xdata"][i], data["ydata"][i], marker=".", label=data["label"] + f" {i + 1}")

        if data["legend"]:
            plt.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(data["title"] + ".png")
        if settings["SHOW-EACH-TEST"]:
            plt.show()
        plt.close("all")

class DataSet:

    def __init__(self, num_groups, max_num_tests):

        # 3-dimensional Dictionary to store the actual data
        self.data_record = {}
        # self.data_record is keyed by group names
        # self.data_record[group] is keyed by tests
        # self.data_record[group][test] is keyed by one of (area, force_max, init_slope, wavelength)

        # Data Means
        self.area_mean_arr = np.zeros(shape=(num_groups, max_num_tests))
        self.force_max_mean_arr = np.zeros(shape=(num_groups, max_num_tests))
        self.init_slope_mean_arr = np.zeros(shape=(num_groups, max_num_tests))
        self.wavelength_mean_arr = np.zeros(shape=(num_groups, max_num_tests))

        # Data STDevs
        self.area_stdev_arr = np.zeros(shape=(num_groups, max_num_tests))
        self.force_max_stdev_arr = np.zeros(shape=(num_groups, max_num_tests))
        self.init_slope_stdev_arr = np.zeros(shape=(num_groups, max_num_tests))
        self.wavelength_stdev_arr = np.zeros(shape=(num_groups, max_num_tests))

        #  # Data Means
        #  self.area_mean_arr = {}
        #  self.force_max_mean_arr = {}
        #  self.init_slope_mean_arr = {}
        #  self.wavelength_mean_arr = {}
        #
        #  # Data STDevs
        #  self.area_stdev_arr = {}
        #  self.force_max_stdev_arr = {}
        #  self.init_slope_stdev_arr = {}
        #  self.wavelength_stdev_arr = {}

    def delRow(self, ind):
        self.area_mean_arr = self.area_mean_arr.tolist()
        del self.area_mean_arr[ind[0]][ind[1]]
        self.area_mean_arr = np.array(self.area_mean_arr)

        self.area_stdev_arr = self.area_stdev_arr.tolist()
        del self.area_stdev_arr[ind[0]][ind[1]]
        self.area_stdev_arr = np.array(self.area_stdev_arr)

        self.force_max_mean_arr = self.force_max_mean_arr.tolist()
        del self.force_max_mean_arr[ind[0]][ind[1]]
        self.force_max_mean_arr = np.array(self.force_max_mean_arr)

        self.force_max_stdev_arr = self.force_max_stdev_arr.tolist()
        del self.force_max_stdev_arr[ind[0]][ind[1]]
        self.force_max_stdev_arr = np.array(self.force_max_stdev_arr)

        self.init_slope_mean_arr = self.init_slope_mean_arr.tolist()
        del self.init_slope_mean_arr[ind[0]][ind[1]]
        self.init_slope_mean_arr = np.array(self.init_slope_mean_arr)

        self.init_slope_stdev_arr = self.init_slope_stdev_arr.tolist()
        del self.init_slope_stdev_arr[ind[0]][ind[1]]
        self.init_slope_stdev_arr = np.array(self.init_slope_stdev_arr)

        self.wavelength_mean_arr = self.wavelength_mean_arr.tolist()
        del self.wavelength_mean_arr[ind[0]][ind[1]]
        self.wavelength_mean_arr = np.array(self.wavelength_mean_arr)

        self.wavelength_stdev_arr = self.wavelength_stdev_arr.tolist()
        del self.wavelength_stdev_arr[ind[0]][ind[1]]
        self.wavelength_stdev_arr = np.array(self.wavelength_stdev_arr)

    def calculateStats(self, test, i):

        self.area_mean_arr[i] = stats.mean(test.area_arr.values())
        self.area_stdev_arr[i] = stats.stdev(test.area_arr.values())

        self.force_max_mean_arr[i] = stats.mean(test.force_max_arr.values())
        self.force_max_stdev_arr[i] = stats.stdev(test.force_max_arr.values())

        self.init_slope_mean_arr[i] = stats.mean(test.init_slope_arr.values())
        self.init_slope_stdev_arr[i] = stats.stdev(test.init_slope_arr.values())

        self.wavelength_mean_arr[i] = stats.mean(test.wavelength_arr.values())
        self.wavelength_stdev_arr[i] = stats.stdev(test.wavelength_arr.values())

    def plotAllRawData(self, group, settings):

        xlabel = "Test"

        values = [
                ("area", f"Area Under the Impulse Curve for {group}", "Area (kN * us)"),
                ("force_max", f"Peak Force for {group}", "Force (kN)"),
                ("init_slope", f"Initial Slope of the Wave for {group}", "Slope (kN / us)"),
                ("wavelength", f"Duration of the Impact Event for {group}", "Duration (us)")
                ]

        for value in values:
            (key, title, ylabel) = value

            data_dict = {}
            for i, (test, dic) in enumerate(list(self.data_record[group].items())):
                temp_data = dic[key]
                for value in dic["rejected_strikes"]:
                    temp_data.pop(value)
                data_dict[i + 1] = temp_data

            data = pd.DataFrame(data=data_dict)
            self.plotData(title, xlabel, ylabel, data, settings, True)

    def plotAllMeanData(self, settings):
        group_list = list(self.data_record.keys())

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
                init_data = deepcopy(self.area_mean_arr)
                min_max_vals = (settings["MEAN-AREA-MIN"], settings["MEAN-AREA-MAX"])
            elif key == "force_max":
                init_data = deepcopy(self.force_max_mean_arr)
                min_max_vals = (settings["MEAN-PEAK-FORCE-MIN"], settings["MEAN-PEAK-FORCE-MAX"])
            elif key == "init_slope":
                init_data = deepcopy(self.init_slope_mean_arr)
                min_max_vals = (settings["MEAN-INIT-SLOPE-MIN"], settings["MEAN-INIT-SLOPE-MAX"])
            elif key == "wavelength":
                init_data = deepcopy(self.wavelength_mean_arr)
                min_max_vals = (settings["MEAN-LENGTH-MIN"], settings["MEAN-LENGTH-MAX"])

            max_len = 0
            for datum in init_data:
                max_len = max(max_len, len(datum))

            data_dict = {}

            for i, datum in enumerate(init_data):
                temp = [0 for _ in range(max_len)]
                for j, d in enumerate(datum):
                    temp[j] = d
                data_dict[group_list[i]] = temp

            data = pd.DataFrame(data=data_dict)

            self.plotData(title, xlabel, ylabel, data, settings, False, min_max_vals)

    def plotData(self, title, xlabel, ylabel, input_data, settings, iterated, resize_axis = None):

        plt.figure(title, figsize=settings["fig_size"])
        plt.grid(True)
        plt.title(title)

        plt.rc("axes", titlesize=settings["TITLE-SIZE"])
        plt.rc("axes", labelsize=settings["LABEL-SIZE"])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        plot = sns.violinplot(data=input_data, inner="point")

        if resize_axis is not None:
            axes = plot.axes
            axes.set_ylim(resize_axis[0], resize_axis[1])

        plt.tight_layout()
        if iterated:
            plt.savefig(f"{title}-{xlabel}.png")
        else:
            plt.savefig(f"{title}.png")
        if settings["SHOW-EACH-FINAL-IMAGE"]:
            plt.show()
        plt.close("all")

