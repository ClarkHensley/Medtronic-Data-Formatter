"""
Updated Medtronic Data Formatter
Maintainer: Clark Hensley (ch3136)
Upstream: https://github.com/ClarkHensley/Medtronic-Data-Formatter
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import statistics as stats
from scipy.special import erfc

from GenerateSettings import formatData


def main():

    # Firstly, generate the settings
    formatData()

    # And read from the data json
    settings = dictFromJson("settings.json")

    # Working Directory of this file
    directory = os.path.dirname(os.path.realpath(__file__))
    DF_directory = os.path.join(directory, "Dataformatter")
    os.chdir(DF_directory)

    if not os.path.exists(os.path.join(DF_directory, "Data")):
        os.mkdir(os.path.join(DF_directory, "Data"))

    if not os.path.exists(os.path.join(DF_directory, "Data/{d}".format(d=settings["DATASET"]))):
        os.mkdir(os.path.join(DF_directory, "Data/{d}".format(d=settings["DATASET"])))

    # Open the relevant files
    runs = dictFromJson("runs.json")
    tests_list = runs[settings["DATASET"]]
    database = dictFromJson("database.json")

    # Get the relevant values out of the "database"

    # Some constants
    ImpulseData1 = np.empty([0, 4])
    ImpulseData2 = np.empty([0, 4])

    Recdata = np.empty([0, 5])

    # Next step is to iterate through each test in the tests list:
    for r, run in enumerate(tests_list):
        print(f"Starting run {run}:")

        run_dict = dict(database[run])

        folder = run_dict["FOLDER"]
        Tests = run_dict["TESTS"]

        num_tests = len(Tests)

        strike_total = 0
        for t, test in enumerate(Tests):
            print(f"Starting test {test} in run {run}:")

            # Confirm the folder exists
            data_dir = os.path.join(directory, "/".join([folder, test]))
            if not os.path.exists(data_dir):
                sys.exit(f"{folder}/{test} Directory was not found")

            # Get each file in the directory
            csvs = os.listdir(data_dir)

            csvs.sort()

            for c, csv in enumerate(csvs):
                data = pd.read_csv(os.path.join(data_dir, csv), delimiter=',', dtype=str).to_numpy(dtype=str)

                if len(data) >= 40000:
                    step = int(len(data) / 20000)
                    data = data[::step]

                accelerometer_present = len(data[0]) == 3

                # TODO This next section is quick-and-dirty. Fix it eventually
                if c == 0:

                    if t == 0:
                        time_units = data[0][0]
                        if time_units == "(s)":
                            time_multiple = 1000000
                        elif time_units == "(ms)":
                            time_multiple = 1000
                        elif time_units == "(us)":
                            time_multiple = 1

                    # Time values for the graphs, based on each CSV
                    time_delta = time_multiple * (float(data[6][0]) - float(data[5][0]))
                    inc = int(settings["timestep"] / (2 * time_delta))
                    ratio = inc / 12000

                    if t == 0:

                        shift = int(settings["TIME-SHIFT"] * (2 * inc / settings["timestep"]))
                        num_rejected_strikes = 0
                        strike_triggered = False

                        # Number of strikes in the CSV
                        strike_count = np.zeros([num_tests])
                        # Storage arrays for time stamps, impact force, and accelerometer values
                        time_arr = np.empty([num_tests, inc * 2 + 2, 100])
                        impact_arr = np.empty([num_tests, inc * 2 + 2, 100])
                        accel_arr = np.empty([num_tests, inc * 2 + 2, 100])
                        # Storage arrays for Area under the curve, force maximum per strike, initial slope, and wavelength
                        area_arr = np.zeros([num_tests, 100])
                        force_max_arr = np.zeros([num_tests, 100])
                        init_slope_arr = np.zeros([num_tests, 100])
                        wavelength_arr = np.zeros([num_tests, 100])

                        # Maintains an Array of True of False if a strike has been rejected
                        rejection_arr = [False for _ in range(100)]
                        # Arrays for the means of the above cumulative arrays
                        area_mean_arr = np.zeros([num_tests])

                        force_max_mean_arr = np.zeros([num_tests])

                        init_slope_mean_arr = np.zeros([num_tests])

                        wavelength_mean_arr = np.zeros([num_tests])
                        # Arrays for the standard deviations of the above cumulative arrays
                        area_stdev_arr = np.zeros([num_tests])
                        force_max_stdev_arr = np.zeros([num_tests])
                        init_slope_stdev_arr = np.zeros([num_tests])
                        wavelength_stdev_arr = np.zeros([num_tests])

                # Constants are handled, time to start analysis
                # Throw out the first 3 values in data, as that's only used for this set-up
                data = data[3:]

                fitting_arr = [False for _ in range(len(data))]

                formatCSV(data, accelerometer_present, settings, fitting_arr)

                # first 0.75% of the data
                for ind, datum in enumerate(data):

                    if ind >= len(data) + 250:
                        break

                    # current slope at this point on the curve
                    curr_slope = (float(data[ind, 1]) - float(data[ind - 1, 1])) / (float(data[ind, 0]) - float(data[ind - 1, 0]))

                    # Determines if the strike has occurred
                    if ind >= inc + 10 and all(data[ind - 250:ind + 250, 1] != "5") and float(data[ind, 1]) >= float(settings["FORCE-LOWER-REASONABLE"] / float(settings["kN"]) and curr_slope >= float(settings["SLOPE-LOWER-LIMIT"])):
                        strike_triggered = True

                        # Gather the waveform data

                        # TODO ??
                        for n_ind in range(len(data[0])):
                            n = ind + n_ind

                            
                            time_arr[t][n_ind][c] = time_multiple * (float(data[n, 0]) - float(data[n_ind, 0]))

                            print("Before:\n", impact_arr[t][n_ind])
                            impact_arr[t][n_ind][c] = float(data[n, 1]) * float(settings["kN"])
                            print("After:\n", impact_arr[t][n_ind])

                            if accelerometer_present:
                                accel_arr[t][n_ind][c] = float(data[n, 2]) / float(settings["mV_to_a"])

                            # Curve-fititng for Force
                            if fitting_arr[ind]:
                                curveFitting(settings, time_arr, impact_arr, time_multiple, inc, n_ind, t, c)

                            # Waveform Characteristics
                            waveformCharacteristics(settings, time_arr, area_arr, force_max_arr, init_slope_arr, impact_arr, wavelength_arr, inc, ratio, shift, time_delta, t, c)

                        # After strike_triggered, we break the inner loop
                        break

                # Now, see if the strike was found in the previous loop
                if not strike_triggered:
                    num_rejected_strikes += 1
                    print(f"Warning: Strike {c + 1} was not triggered!")

                else:
                    print(f"Srike {c + 1} accepted!")
                    strike_count[t] = c
                    strike_triggered = False

            # Statistical Analysis for each run
            final_strike_ind = 0

            for f in range(int(strike_count[t]) + 1):

                if force_max_arr[t][f] >= settings["FORCE-LOWER-REASONABLE"] and force_max_arr[t][f] <= settings["FORCE-UPPER-REASONABLE"] and area_arr[t][f] >= settings["AREA-LOWER-LIMIT"]:
                    area_arr[final_strike_ind] = area_arr[f]
                    force_max_arr[final_strike_ind] = force_max_arr[f]
                    init_slope_arr[final_strike_ind] = init_slope_arr[f]
                    wavelength_arr[final_strike_ind] = wavelength_arr[f]
                    rejection_arr[f] = False
                    final_strike_ind += 1

                else:
                    rejection_arr[f] = True

            # Calculate statistics
            selected_area_arr = area_arr[0:final_strike_ind]
            selected_force_max_arr = force_max_arr[0:final_strike_ind]

            area_mean_arr[t] = stats.mean(selected_area_arr)
            area_stdev_arr[t] = stats.stdev(selected_area_arr, area_mean_arr[t])
            force_max_mean_arr[t] = stats.mean(selected_force_max_arr)
            force_max_stdev_arr[t] = stats.stdev(selected_force_max_arr, force_max_mean_arr[t])
            init_slope_mean_arr[t] = stats.mean(init_slope_arr[0:final_strike_ind])
            init_slope_stdev_arr[t] = stats.stdev(init_slope_arr[0:final_strike_ind], init_slope_mean_arr[t])
            wavelength_mean_arr[t] = stats.mean(wavelength_arr[0:final_strike_ind])
            wavelength_stdev_arr[t] = stats.stdev(wavelength_arr[0:final_strike_ind], wavelength_mean_arr[t])

            # Reject Values based on "Chauvenets and absurdly low values"
            rejected_area_bool = ((erfc(np.abs(selected_area_arr) / area_stdev_arr[t])) > (1 / (2 * len(selected_area_arr))))
            rejected_force_max_bool = ((erfc(np.abs(selected_force_max_arr) / force_max_stdev_arr[t])) > (1 / (2 * len(selected_force_max_arr))))

            init_rejection_ind = 0
            chauv_rejection_ind = 0

            for i in range(0, final_strike_ind):
                if rejection_arr[i]:
                    init_rejection_ind += 1

                if rejected_area_bool[i] and rejected_force_max_bool[i]:
                    area_arr[t][chauv_rejection_ind] = area_arr[t][i]
                    init_slope_arr[t][chauv_rejection_ind] = init_slope_arr[t][i]
                    wavelength_arr[t][chauv_rejection_ind] = wavelength_arr[t][i]
                    rejection_arr[i + init_rejection_ind] = False

                    chauv_rejection_ind += 1

                    ImpulseData1 = np.vstack([ImpulseData1, [area_arr[t][i],  force_max_arr[t][i], init_slope_arr[t][i], wavelength_arr[t][i]]])

                else:
                    print("Rejected Strike: {s}".format(s=i+chauv_rejection_ind+init_rejection_ind))
                    rejection_arr[i + rejected_area_bool] = True

            for i in range(chauv_rejection_ind):

                if not rejection_arr[i]:
                    # TODO should probably be a function
                    plt.figure("ForcePlot", figsize=(18, 10))
                    plt.grid(True)
                    plt.tight_layout()
                    plt.title(Tests[t])
                    plt.ylabel("Force (kN)")
                    plt.xlabel("Time (us)")

                    plt.rc("axes", titlesize=settings["TITLE-SIZE"])
                    plt.rc("axes", labelsize=settings["LABEL-SIZE"])
                    plt.plot(time_arr[t, 1:inc * 2 - 5, i], force_max_arr[t, 1:inc * 2 - 5, i], marker=".", label=f"Strike {str(i + 1)}")
                    if settings["LEGEND"]:
                        plt.legend(loc="upper left")
                    plt.axis([0, 300, 0, 25])
                    plt.savefig(os.path.join(DF_directory, "Data/{d}/{t}-ForcePlot.png".format(d=settings["DATASET"], t=test)))
                    plt.close("ForcePlot")

                    if accelerometer_present:
                        # waveplot(1, strike, test, inc * 2 - 5, Testname[test], "Time (us), "Acceleration (m/s^2), 1, legendOn)
                        plt.figure("AccelPlot", figsize=(18, 10))
                        plt.grid(True)
                        plt.tight_layout()
                        plt.title(Tests[t])
                        plt.ylabel("Acceleration (m/s^2)")
                        plt.xlabel("Time (us)")
                        plt.rc("axes", titlesize=settings["TITLE-SIZE"])
                        plt.rc("axes", labelsize=settings["LABEL-SIZE"])
                        plt.plot(time_arr[t, 1:inc * 2 - 5, c], accel_arr[t, 1:inc * 2 - 5, c], marker=".", label="Strike")
                        if settings["LEGEND"]:
                            plt.legend(loc="upper left")
                        plt.axis([0, 300, -9, 7])
                        plt.savefig(os.path.join(DF_directory, "Data/{d}/{t}-AccelPlot.png".format(d=settings["DATASET"], t=test)))
                        plt.close("AccelPlot")

            # Plot the Peak Force Data
            plt.figure("PeakForcePlot", figsize=(18, 10))
            plt.grid(True)
            plt.tight_layout()
            plt.title(Tests[t])
            plt.ylabel("Force (kN)")
            plt.xlabel("Stike Count")
            # 2
            plt.plot(list(range(1, chauv_rejection_ind)), force_max_arr[t, 0:inc * 2 - 5], marker=".")

            plt.axis([0, 300, 0, 25])
            plt.savefig(os.path.join(DF_directory, "Data/{d}/{t}-PeakForcePlot".format(d=settings["DATASET"], t=test)))
            plt.close("PeakForcePlot")

            # Update Statistics
            area_mean_arr[t] = stats.mean(area_arr[0:chauv_rejection_ind])
            area_stdev_arr[t] = stats.stdev(area_arr[0:chauv_rejection_ind], area_mean_arr[t])

            force_max_arr[t] = stats.mean(force_max_arr[0:chauv_rejection_ind])
            force_max_stdev_arr = stats.stdev(force_max_arr[0:chauv_rejection_ind], force_max_mean_arr[t])

            init_slope_mean_arr = stats.mean(init_slope_arr[0:chauv_rejection_ind])
            init_slope_stdev_arr = stats.stdev(init_slope_arr[0:chauv_rejection_ind], init_slope_mean_arr[t])

            wavelength_mean_arr[t] = stats.mean(wavelength_arr[0:chauv_rejection_ind])
            wavelength_stdev_arr[t] = stats.stdev(wavelength_arr[0:chauv_rejection_ind], wavelength_mean_arr[t])

            # Increment strike total
            strike_total += strike_count[t] + 1

        # Outside of while Test
        final_names = []
        for i in range(num_tests):
            ImpulseData2 = np.vstack([ImpulseData2, [area_mean_arr[i], force_max_mean_arr[i], init_slope_mean_arr[i], wavelength_mean_arr[i]]])

            final_name = run
            if settings["SHOW-STRING-COUNT"]:
                final_name += " " + str(round(strike_total / num_tests, 1))
            if settings["SHOW-N-COUNT"]:
                final_name += "n = " + str(num_tests)

            final_names.append(final_name)

            Recdata = np.vstack([Recdata, [final_name, area_mean_arr[i], force_max_mean_arr[i], init_slope_mean_arr[i], wavelength_mean_arr[i]]])

        # Plot the Data
        data_columns = np.array(["Area", "Force", "Slope", "Wavelength"])
        name_columns_test = np.array(["Test"])

        ImpulseDataFrame1 = pd.DataFrame(ImpulseData1, columns=data_columns)
        TestNameFrame = pd.DataFrame(final_names, columns=name_columns_test)

        createChart("Area", TestNameFrame.Test, ImpulseDataFrame1.Area, f"Area Under the Impulse curve {run}", "Test Name", "Area (kN * us)", settings, DF_directory)

        createChart("Max Force", TestNameFrame.Test, ImpulseDataFrame1.Force, f"Peak Force {run}", "Test name", "force (kN)", settings, DF_directory)

        createChart("Slope", TestNameFrame.Test, ImpulseDataFrame1.Slope, f"Iniital Slope of the Wave {run}", "Test Name", "Slope (kN / us)", settings, DF_directory)

        createChart("Wavelength", TestNameFrame.Test, ImpulseDataFrame1.Wavelength, f"Duration of the Impact Event {run}", "Test Name", "Duration (us)", settings, DF_directory)

    name_columns_group = np.array(["Group"])
    ImpulseDataFrame2 = pd.DataFrame(ImpulseData2, columns=data_columns)
    GroupNameFrame = pd.DataFrame(tests_list, columns=name_columns_group)

    createChart("Cumulative Area", GroupNameFrame.Group, ImpulseDataFrame2.Area, "Area Under the Impulse Curve", "Group", "Area (Kn * us)", settings, DF_directory)

    createChart("Cumulative Max Force", GroupNameFrame.Group, ImpulseDataFrame2.Force, "Peak Force", "Group", "Force (kN)", settings, DF_directory)

    createChart("Cumulative Initial Slope of the Wave", GroupNameFrame.Group, ImpulseDataFrame2.Slope, "Initial Slope of the Wave", "Group", "Slope (kN / us)", settings, DF_directory)

    createChart("Cumulative Duration of the Impact Event", GroupNameFrame.Group, ImpulseDataFrame2.Wavelength, "Duration of the Impace Event", "Group", "Duration (us)", settings, DF_directory)

    # Finally, record values to a new CSV
    new_path = os.path.join(DF_directory, "Data/{d}/{d}-Recorded-CSV".format(d=settings["DATASET"]))
    records_column = np.array(["Implant", "Area", "Force", "Slope", "Wavelength"])

    record_data_frame = pd.DataFrame(Recdata, columns=records_column)
    record_data_frame.to_csv(new_path, encoding="utf-8")


def dictFromJson(file):
    """ Attempt to open a .json file, which will be converted into a python Dictionary."""

    # Attempt to create a dictionary from the file and return it
    try:
        with open(file, "r") as h:
            h_content = h.read()
            temp = json.loads(h_content)
        return temp
    except FileNotFoundError:
        sys.exit("{missing_file} JSON file could not be found.".format(missing_file=file))


def formatCSV(data, accelerometer_present, settings, fitting_arr):
    """
    Remove invalid values from the CSV, take care of under- or over-flow errors
    """
    # Data[0, 1] should always be 0, for some reason
    data[0, 1] = 0

    for d, datum in enumerate(data):
        if datum[1] == "âˆž" or datum[1] == "∞":
            data[d, 1] = "5"
            fitting_arr[d] = True

        if datum[1] == "-∞" or float(datum[1]) < 0:
            data[d, 1] = "0"
            fitting_arr[d] = True

        if accelerometer_present:
            if datum[2] == "âˆž" or datum[2] == "∞":
                data[d, 2] = str(settings["max_av"])

            if datum[2] == "-∞":
                data[d, 2] = str(-1 * settings["max_av"])

def curveFitting(settings, time_arr, impact_arr, time_multiple, inc, n_ind, t, c):
    #delta = (time_arr[t][2][c] - time_arr[t][1][c]) * time_multiple

    start_interpolation = False

    for i in range(0, 2 * inc - 1):
        check_val = abs(impact_arr[t][i + 1][c] - impact_arr[t][i][c]) #/ delta
        if not start_interpolation:
            if check_val >= 10:
                start_interpolation = True

                # start_time is always offset by residue
                start_time = i - float(settings["residue"])
        else:
            if check_val <= 2 and impact_arr[t][i][c] >= 0 and impact_arr[t][i][c] < 4 * float(settings["kN"]):
                start_interpolation = False
                end_time = i


                interpolation_slope = (impact_arr[t][end_time + 1][c] - impact_arr[t][start_time][c]) / (time_arr[t][end_time][c] - time_arr[t][start_time][c])
                for j in range(start_time, end_time):
                    impact_arr[t][j][c] = impact_arr[t][j - 1][c] + interpolation_slope * delta

                # TODO eww
                i = j

            if i == 2 * inc - 2:
                end_time = 2 * inc - 1
                interpolation_slope = (impact_arr[t][end_time + 1][c] - impact_arr[t][start_time][c]) / (time_arr[t][end_time][c] - time_arr[t][start_time][c])
                
                for j in range(start_time, end_time + 1):
                    impact_arr[t][j][c] = impact_arr[t][j - 1][c] + interpolation_slope * delta

                # TODO eww
                i = j

        if impact_arr[t][n_ind][c] < float(settings["FORCE-LOWER-BOUND"]) * float(settings["kN"]):
            impact_arr[t][n_ind][c] = float(settings["FORCE-LOWER-BOUND"]) * float(settings["kN"])


def waveformCharacteristics(settings, time_arr, area_arr, force_max_arr, init_slope_arr, impact_arr, wavelength_arr, inc, ratio, shift, time_delta, t, c):
    """
    Format the data for the waveforms
    """

    init_slope_point_cap = int(500 * ratio)
    wave_duration = False

    for i in range(0, 2 * inc):
        area_arr[t][c] = area_arr[t][c] + (impact_arr[t][i][c] * time_delta)

        if force_max_arr[t][c] < impact_arr[t][i][c]:
            force_max_arr[t][c] = impact_arr[t][i][c]

        if i <= (inc + init_slope_point_cap / 2 - 1 - shift) and i > inc - init_slope_point_cap / 2 - shift:
            #print(t, i, c)
            #print(impact_arr[t, i])
            #print(time_arr[t, i])
            #print(impact_arr[t, i - 1])
            #print(time_arr[t, i - 1])
            init_slope_arr[t][c] += ((impact_arr[t][i][c] - impact_arr[t][i - 1][c]) / (time_arr[t][i][c] - time_arr[t][i - 1][c])) / init_slope_point_cap

        if not wave_duration and (impact_arr[t][i][c] >= 3):
            wave_duration = True
            ws = i
        elif wave_duration and (impact_arr[t][i][c] <= settings["WAVE-END"]):
            wavelength_arr[t][c] = (i - ws) * (settings["timestep"] / (inc * 2))
            wave_duration = False


def createChart(name, x_data, y_data, t_header, x_label, y_label, settings, DF_directory):
    """
    Create a PLT chart of the data
    """

    plt.figure(name, figsize=(18, 10))
    plt.title(t_header)

    plt.rc("axes", titlesize=settings["TITLE-SIZE"])
    plt.rc("axes", labelsize=settings["LABEL-SIZE"])

    if settings["PLOT-TYPE"] == "VIOLIN":
        sns.violinplot(x=x_data, y=y_data)
        filename = t_header + "-Violin-Plot.png"
    elif settings["PLOT-TYPE"] == "BAR":
        plt.boxplot(x=x_data, y=y_data)
        filename = t_header + "-Bar-Graph.png"

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    if not os.path.exists(os.path.join(DF_directory, "Data/{d}".format(d=settings["DATASET"]))):
        os.mkdir(os.path.join(DF_directory, "Data/{d}".format(d=settings["DATASET"])))
    print(os.path.join(DF_directory, "Data/{d}/{fn}".format(d=settings["DATASET"], fn=filename)))
    plt.savefig(os.path.join(DF_directory, "Data/{d}/{fn}".format(d=settings["DATASET"], fn=filename)))
    plt.close(name)


if __name__ == "__main__":
    main()

