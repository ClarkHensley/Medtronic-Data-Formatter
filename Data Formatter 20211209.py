import numpy as np #Used for array operations
import matplotlib.pyplot as plt #Used to plot scatter plots of waveform data
import pandas as pd #Used to read csv files
import seaborn as sns #Used for violin plots
import time #Used for runtime optimization purposes
#import pathlib #Used for checking if a file exists or not
import os #Used for checking if a file exists or not
import sys #Used to exit the program if a known error occurs
import statistics as stats #Used to calculate stdev and mean values from nparrays
from scipy.special import erfc #Used for chauvenets criterion
import getpass #Used to find the current user's username


def main():
    """
    Main Function for Medtronic Data Formatter
    Maintainer: Clark Hensley (ch3136)
    Upstream: https://github.com/ClarkHensley/Medtronic-Data-Formatter
    """

    #####
    #
    # Constant variables
    #
    #####

    # Dataset being tested
    # TODO take input from a config file (json)
    curr_dataset = "Medtronic Data 20210806" #run name file 
    #put all excel file folders in ''C:\Users\[user]\Documents\Waveforms\Medronic Tests 20210806'' for RunName = "Medtronic Data 20210806"
    # -Current Run Names-
    # "20210804 Medtronic Manuscript Tests" 
    # "Medtronic Data 20210806"
    # "20210804 Medtronic Manuscript Tests alt" 
    
    # Output format settings
    # TODO take input from a config file
    text_fontsize = 18
    label_fontsize = 22;
    title_fontsize = 30;
    # TODO ??
    plt.rcParams.update({'font.size': int(text_fontsize)}) #Sets the fonts for plots 
    plottype = 0 #Violin(0) or Bar Graph(1)
    NoiseLevel = 0.1 #Value used if noisy data occurs. Provides a lower value for force values. Only used when curve fitting occurs
    NoiseLimit = 0.1  #Value used if noisy data occurs. Provides a lower limit for force values. Only used when curve fitting occurs
    Show_str = 0 #are average string counts shown
    Show_n = 0 #are n = [testcount] shown
    legendOn = 0 #is the legend shown for waveplots
    timeshift = 75 #amount shifted on the x axis for optimal plot viewing
    
    # Project-Specific Constants
    # TODO store in a defaults/constants file
    Flimit = 3 #Limit for how low the Max Force value can be and still be reasonable
    Flimit2 = 20 #Upper bound limit for a reasonable peak force value
    Alimit = 250 #Limit for how low the Area value can be and still be reasonable
    Slimit = 0 #Limit for how low the slope value can be and still be reasonable
    waveend = 1.5 #What force value is considered below the threshold for the impact duration to be considered finished 

    # Universal Constants
    # TODO store in a defaults/constants file
    kN = 4.44822162 #converts voltage to lbf to kN (1V = 1000lbf for our sensor)
    mVtoa = 1.090 #converts milli-volts to acceleration (1V = 1000lbf for our sensor)
    Maxav = 10 #What the picoscope range for acceleration was set to (Should stay the same unless another accelerometer is used)

    # Working Directory of this file, all paths will be relative to this:
    directory = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(directory, "Dataformatter")

    # Check if the relevant folders and settings files exist. If not, either exit or create them 
    ensure_folders(directory, curr_dataset)

    # Open the actual file of the test
    group_names, group_abbr = ""
    with open(os.path.join(directory, f"{curr_dataset}.ini"), "r") as h:
        file_data = h.read()
        data_parts = file_data.split(", $")
        group_names = data_parts[0].split(", ")
        group_abbr = data_parts[1].split(", ")

    # Violin Chart params and constants
    # TODO Constants or remove? Not sure if this initialization is necessary
    ImpulseData = np.empty([0,4]) #collects the maxforce, area, initial slope, and impact duration between tests
    TestNameData = np.empty(0) #collects the testnames to be compared. Uses Abrevation names
    ImpulseData2 = np.empty([0,4]) #collects the maxforce, area, initial slope, and impact duration between groups
    GroupNameData = np.empty(0) #collects the groupnames to be compared
    Recdata = np.empty([0,5])

    # Now is the right time to parse the "database"
    database = {}
    with open(os.path.join(directory, "database.ini")) as db:
        raw_data = db.read()
        split_data = raw_data.split("$")
        del split_data[-1]
        for d in split_data:
            d = d.strip()
            db_name, db_data = d.split("\n")
            database[db_name] = db_data

    num_tests = len(group_names)

    for g, group in enumerate(group_names):
        # Main Loop
        try:
            params = database[group]

            ind = int(3/2 + len(params) / 2)

            Folder = params[0] #Test Group Name      
            ManuCount = float(params[1]) #Manualy imputted average strike count for the group
            cond1 = bool(int(params[2])) #Should the trigger point be closer torward the peak (usually makes the plot look better) 
            TestName = params[3:ind]
            Abrv = params[ind:]

        except KeyError:
            sys.exit(f"Missing Configuration Parameters for {group}")

        total_num_strikes = 0;
        for t in range(num_tests):
            print(f"Starting Test: {TestName[t]}")

            data_dir = os.path.join(directory, f"{Folder}/{TestName[t]}")
            if os.path.exists(data_dir):
                data_csvs = [name for name in os.listdir(data_dir) if os.path.isfile(name)]
            else:
                sys.exit(f"Missing Data for Test: {TestName[t]}")

            strike_ind = 0

            # Open the first file manually to get relevant data
            with pd.read_csv(data_csvs[0], delimiter=",", dtype=str) as datacsv:
                data = datacsv.to_numpy(dtype=str)

            if(len(data) >= 40000):
                data = data[::len(data)/20000]

            # Determine if accelerometer data exists
            accelerometer = len(data[0] == 3)

            mult = 1
            if data[0][0] == "(s)":
                mult = 1E6
            elif data[0][0] == "(ms)":
                mult = 1E3

            # Time to capture in microseconds
            timerange = 300
            # Amount of time between each reading
            del_t = float(data[6][0]) - float(data[5][0])
            # interval before and after the trigger event
            inc = int(timerange / (2 * del_t))
            # Changes the sampling rate
            ratio = inc / 12000
            # Shift of the view of the data
            shift = int(timeshift * (inc * 2 / timerange))


            # TODO I don't understand these
            strikes_rejected = 0
            strike_triggered = False
            t_rec, F_rec, a_rec = np.empty([num_tests, (2 * inc) + 2, 100])

            Area, Fmax, IniSlope, Wavelength = np.zeros([num_tests, 100])

            Rejbool = np.zeros([100])

            Areamean, MaxFmean, IniSlpoemean, Wavelengthmean, AreaSTDev, MaxFSTDev, IniSlopeSTDev, WavelengthSTDev = np.zeros([num_tests])

            # At this point, we can slice the front off the data
            data = data[3:]

            # TODO why?
            data[0, 1] = 0

            # Determine if "curve-fitting is necessary"
            for i in range(int(0.0075 * len(data))):
                useFitting = fittingFunc(data)
                slopecheck = (float(data[i, 1]) - float(data[i - 1, 1])) / (float(data[i, 0]) - float(data[i - 1, 0]))
                
            # Formats more voltage
            # TODO !!!!!!!!
            if i >= inc + 10 and float(data[i, 1]) >= Flimit/kN and slopecheck >= Slimit and not "5" in data[i-250:i+250, 1]:
                striktriggered = True
                start = i - inc + shift
                stop = i + inc + shift
                for n in range(start, stop):
                    if n < int(len(data)) - 1:
                        array_count = int(n - start)

                        t_rec[t][array_count][strike]

                    else:
                        break




            







def ensure_folders(directory, curr_dataset):
    """
    Check if the relevant folders exist
    """

    # Ensure the configuration files exist
    # TODO move these to a sensible format, maybe delete them
    # What are they for?
    if not os.path.exists(os.path.join(directory, f"{curr_dataset}.ini")):
        sys.exit(f"Error: {curr_dataset}.ini file is not found!\n")

    if not os.path.exists(os.path.join(directory, "database.ini")):
        sys.exit("Error: database.ini file is not found!\n")

    if not os.path.exists(os.path.join(directory, f"Data/{curr_dataset}")):
        os.makedirs(f"Data/{curr_dataset}")


def chart(directory, curr_dataset, fignum, xarray, yarray, tname, yname, xname, pltt, label_fontsize, title_fontsize):
    plt.figure(int(fignum),figsize=(18,10)); plt.title(tname)
    plt.rc('axes', titlesize=title_fontsize) #fontsize of the title
    plt.rc('axes', labelsize=label_fontsize) #fontsize of the x and y labels

    if pltt == 0:
        sns.violinplot(x=xarray, y=yarray)
        filename = tname + "-violin-graph.png"
    else:
        # TODO ???
        #plt.boxplot(x=xarray, y=yarray)
        plt.boxplot(x=xarray, labels=yarray)
        filename = tname + "-box-graph.png"

    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"Dataformatter/Data/{curr_dataset}/{filename}"))

    plt.show()
    
#Plots waveform scatterpolts    
def waveplot(fignum,strnum,testnum,incnum,tname,xname,yname,pltt,legend, title_fontsize, label_fontsize): 
    Count = list(range(0,101))
    plt.figure(int(fignum),figsize=(18,10))
    plt.grid(True)
    plt.tight_layout()
    plt.title(tname)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.rc('axes', titlesize=title_fontsize) #fontsize of the title
    plt.rc('axes', labelsize=label_fontsize) #fontsize of the x and y labels
    if pltt == 0:
        plt.plot(t_rec[testnum,1:incnum,strnum], F_rec[testnum,1:incnum,strnum], marker='.', label = "Strike " + str(strnum + 1))   
    elif pltt == 1:
        plt.plot(t_rec[testnum,1:incnum,strnum], a_rec[testnum,1:incnum,strnum], marker='.', label = "Strike " + str(strnum + 1))   
    elif pltt == 2:
        plt.plot(Count[1:incnum+1], Fmax[testnum,0:incnum], marker='.')
    if legend == 1:
        plt.legend(loc = "upper left")

#Creates an array only containing information to be used in statistical analysis
def statarray(snum,stattype):
    selected = np.empty(0)
    if stattype == 0:
        for j in range(0, snum):
            selected = np.append(selected, Area[test][j])
    if stattype == 1:
        for j in range(0, snum):
            selected = np.append(selected, Fmax[test][j])
    if stattype == 2:
        for j in range(0, snum):
            selected = np.append(selected, IniSlope[test][j])
    if stattype == 3:
        for j in range(0, snum):
            selected = np.append(selected, Wavelength[test][j])
    return selected

#Formats a group name for the analysis charts
def NamingFormat(gnum, snum, tnum, count, incstr, incn):
    gname = GroupAbrv[gnum]
    if incstr == 1:
        Avgstr = str(count)
        gname += " " + Avgstr + " str"
    if incn == 1:
        gname += " n = " + str(tnum)
    return gname

#Determines whether a data point is reasonable or not (gets rid of outliers)
def chauvenetscrit(x,datalength,datamean,datastdev):
    tx = np.abs(x - datamean)/datastdev                       #distance from mean in stdev                                        
    chauvcrit = 1/(2*datalength)                              #rejection criterion
    Pt = erfc(tx)                                             #Probability function
    return Pt < chauvcrit                                     #reject/keep data value

#Reads ini files and finds if a match occurs
def filefindmatch(strmatch, CF):
    ch = "" #ch is use to read characters from a txt file. Strings {$ , ~ : *} are used to activate certain commands 
    while ch != "*":
        ch = file.read(1)
        if ch == ":":
            word = ""
            ch = file.read(1)
            while ch != "~":
                word = word + ch
                ch = file.read(1)
            if word == strmatch:
                ch = file.read(1)
                if CF == 1:
                    file.close()
                return 1
    if CF == 1:
        file.close()
    return 0

#Reads ini files and returns an array of values or a single value
def filereader(CF):
    ch = file.read(1) #ch is use to read characters from a txt file. Strings {$ , ~ : *} are used to activate certain commands 
    devarray = np.array([""]) #Array made up of the group names analysed in this run through
    while ch != "$":
        word = "" #word collects ch charcters into words from the txt file read
        while ch != ",":
            word = word + ch
            ch = file.read(1)
        if word != "":
            if devarray[0] == "":
                devarray = word
            else:
                devarray = np.append(devarray, word)
        ch = file.read(1)
        ch = file.read(1)
    ch = file.read(1)
    if CF == 1:
        file.close()
    return devarray
    
def final_plots():
    'Plotting statistical analysis violin charts by group'  
    DataCOL2 = np.array(["Area", "Force", "Slope","Length"]); NameCOL2 = np.array(["Group"])       
    ImpulseDataframe2 = pd.DataFrame(ImpulseData2, columns = DataCOL2); GroupNameFrame = pd.DataFrame(GroupNameData, columns = NameCOL2)      
    chart(3,GroupNameFrame.Group,ImpulseDataframe2.Area,"Area Under the Impulse Curve","Area (kN*us)","Group",plottype)
    chart(4,GroupNameFrame.Group,ImpulseDataframe2.Force,"Peak Force","Force (kN)","Group",plottype)
    chart(5,GroupNameFrame.Group,ImpulseDataframe2.Slope,"Initial Slope of the Wave","Slope (kN/us)","Group",plottype)
    chart(6,GroupNameFrame.Group,ImpulseDataframe2.Length,"Duration of the Impact Event","Duration (us)","Group",plottype)
    plt.close("all")
        
    'Writing Average Test Parameter Values to csv'    
    filename = "C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//Recorded Data.csv"
    RecCOL = np.array(["Implant","Area", "Force", "Slope","Length"])
    #Recdata = np.vstack((GroupNameData,ImpulseData2))      
    RecDataframe = pd.DataFrame(Recdata, columns = RecCOL)
    RecDataframe.to_csv(filename, encoding='utf-8')


def fittingFunc(data):
    """
    This function serves two purposes. If the data are listing an overflow or underflow of voltage, fix it. Also, if any such fix is required, return True to set the fitting boolean, else return false
    """
    fitting_bool = False
    for x in range(len(data)):
        overflow_check = ["âˆž", "∞"]
        underflow_check = "-∞"
        if data[x, 1] in overflow_check:
            data[x, 1] = "5"
            if not fitting_bool:
                fitting_bool = True
    
        if data[x, 1] == underflow_check or float(data[x, 1]) < 0:
            data[x, 1] = "0"
            if not fitting_bool:
                fitting_bool = True
    
    return fitting_bool


            
