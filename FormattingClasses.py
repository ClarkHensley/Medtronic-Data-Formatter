import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

from copy import deepcopy



"""
Classes for formatting Medtronic Data
"""


class StrikeSet:
    """
    This class just holds data for each CSV in one place
    """

    def __init__(self, data, fitting_arr, group, time_storage, strike, settings):

        self.group = group
        self.name = strike

        self.fitting_arr = fitting_arr

        self.data = data
        self.accelerometer_present = len(data[0]) == 3
        self.timedelta = round(float(data[1, 0]) - float(data[0, 0]), 2)

        self.inc = int(settings["timestep"]) / (2 * self.timedelta)

        self.arrsize = 2 * int(self.inc) + 2

        self.ratio = self.inc / 12000

        self.time_multiple = 0

        if(time_storage == "(s)"):
            self.time_multiple = 1000000
        elif(time_storage == "(ms)"):
            self.time_multiple = 1000
        elif(time_storage == "(us)"):
            self.time_multiple = 1

        self.time_arr = np.zeros(shape=(self.arrsize))
        self.impact_arr = np.zeros(shape=(self.arrsize))
        self.accel_arr = np.zeros(shape=(self.arrsize))

        self.rejection_arr = np.array([False for _ in range(self.arrsize)])

        self.strike_triggered = False

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def fitCurve(self, settings):

        start_interpolation = False
        delta = (self.time_arr[1] - self.time_arr[0]) * self.time_multiple
        print(f"Delta: {delta}")
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
                if abs(next_val - previous_val) / delta >= 10:
                    start_interpolation = True
                    start_index = ind - settings["residue"]

            if self.impact_arr[self.arrsize - 3] < check_val:
                self.impact_arr[self.arrsize - 3] = check_val



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

        # Total Data Values
        self.total_time_arr = None
        self.total_impact_arr = None
        self.total_accel_arr = None

        # Calculated Values
        self.area_arr = np.zeros(shape=(dim))
        self.force_max_arr = np.zeros(shape=(dim))
        self.init_slope_arr = np.zeros(shape=(dim))
        self.wavelength_arr = np.zeros(shape=(dim))

    def __getitem__(self, key):
        return self.strike_set[key]

    def __setitem__(self, key, value):
        self.strike_set[key] = value

    def addStrikeSet(self, strike_set, ind, settings):

        if self.strike_set is not None:
            del self.strike_set
            self.strike_set = None
        
        self.current_ind = ind

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
    
    def finalize(self):

        if len(self.rejected_strikes) != 0:

            self.area_arr = np.delete(self.area_arr, self.rejected_strikes)
            self.force_max_arr = np.delete(self.force_max_arr, self.rejected_strikes)
            self.init_slope_arr = np.delete(self.init_slope_arr, self.rejected_strikes)
            self.wavelength_arr = np.delete(self.wavelength_arr, self.rejected_strikes)

    def getCurrentSlope(self, ind):

        return (float(self.strike_set.data[ind, 1]) - float(self.strike_set.data[ind - 1, 1])) / (float(self.strike_set.data[ind, 0]) - float(self.strike_set.data[ind - 1, 0]))

    def characterizeWaveform(self, settings, s):

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
                if self.strike_set.impact_arr[ind] <= float(settings["WAVE-END"]):
                    self.wavelength_arr[s] = (ind - wave_start) * (settings["timestep"] / (2 * self.strike_set.inc))
                    wave_duration = False

            else:
                if self.strike_set.impact_arr[ind] >= 3:
                    wave_duration = True
                    wave_start = ind

    def plotAllData(self, settings):

        # Plot Time v. Force
        time_v_force_data = {
                "title": f"Time v. Force for {self.name}",
                "xlabel": "Time (us)",
                "ylabel": "Force (kN)",
                "xdata": np.ndarray(shape=(self.strike_count, self.strike_set.arrsize)),
                "ydata": np.ndarray(shape=(self.strike_count, self.strike_set.arrsize)),
                "label": "Strike",
                "legend": settings["LEGEND"]
                }

        # Plot Time v. Accel
        time_v_accel_data = {
            "title": f"Time v. Accel for {self.name}",
            "xlabel": "Time (us)",
            "ylabel": "Acceleration (m/s^2)",
            "xdata": np.ndarray(shape=(self.strike_count, self.strike_set.arrsize)),
            "ydata": np.ndarray(shape=(self.strike_count, self.strike_set.arrsize)),
            "label": "Strike",
            "legend": settings["LEGEND"]
            }

        # Plot Strike v. Force
        strike_v_max_force_data = {
                "title": f"Strike Number v. Max Force for {self.name}",
                "xlabel": "Strike Number",
                "ylabel": "Force (kN)",
                "xdata": np.array(list(self.strike_sets.keys())),
                "ydata": self.force_max_arr,
                "label": "",
                "legend": False
                }

        for s, strike in enumerate(list(self.strike_sets.values())):

            time_v_force_data["xdata"][s] = strike.time_arr
            time_v_force_data["ydata"][s] = strike.impact_arr

            if strike.accelerometer_present:
                time_v_accel_data["xdata"][s] = strike.time_arr
                time_v_accel_data["ydata"][s] = strike.accel_arr
    
        self.plotData(time_v_force_data, settings)

        self.plotData(strike_v_max_force_data, settings)

        if strike.accelerometer_present:
            self.plotData(time_v_accel_data, settings)

    def plotData(self, data, settings):

        plt.figure(data["title"])
        plt.grid(True)
        plt.tight_layout()
        plt.title(data["title"])
        plt.xlabel(data["xlabel"])
        plt.ylabel(data["ylabel"])

        plt.rc("axes", titlesize=settings["TITLE-SIZE"])
        plt.rc("axes", labelsize=settings["LABEL-SIZE"])

        if data["xdata"].ndim == 1 and data["ydata"].ndim == 1:
            plt.plot(data["xdata"], data["ydata"], marker=".", label=data["label"] + " 1")

        else:
            for i, _ in enumerate(data["xdata"]):
                plt.plot(data["xdata"][i], data["ydata"][i], marker=".", label=data["label"] + f" {i + 1}")

        if data["legend"]:
            plt.legend(loc="upper left")

        plt.show()


class DataSet:

    def __init__(self, dim1, dim2):

        # Data Means
        self.area_mean_arr = np.zeros(shape=(dim1, dim2))
        self.force_max_mean_arr = np.zeros(shape=(dim1, dim2))
        self.init_slope_mean_arr = np.zeros(shape=(dim1, dim2))
        self.wavelength_mean_arr = np.zeros(shape=(dim1, dim2))

        # Data STDevs
        self.area_stdev_arr = np.zeros(shape=(dim1, dim2))
        self.force_max_stdev_arr = np.zeros(shape=(dim1, dim2))
        self.init_slope_stdev_arr = np.zeros(shape=(dim1, dim2))
        self.wavelength_stdev_arr = np.zeros(shape=(dim1, dim2))

        # Keeps track of all strikes
        self.strike_record = {}
        self.strike_count = {}

    def calculateStats(self, test, i):

        self.area_mean_arr[i] = stats.mean(test.area_arr)
        self.area_stdev_arr[i] = stats.stdev(test.area_arr, self.area_mean_arr[i])

        self.force_max_mean_arr[i] = stats.mean(test.force_max_arr)
        self.force_max_stdev_arr[i] = stats.stdev(test.force_max_arr, self.force_max_mean_arr[i])

        self.init_slope_mean_arr[i] = stats.mean(test.init_slope_arr)
        self.init_slope_stdev_arr[i] = stats.stdev(test.init_slope_arr, self.init_slope_mean_arr[i])

        self.wavelength_mean_arr[i] = stats.mean(test.wavelength_arr)
        self.wavelength_stdev_arr[i] = stats.stdev(test.wavelength_arr, self.wavelength_mean_arr[i])

