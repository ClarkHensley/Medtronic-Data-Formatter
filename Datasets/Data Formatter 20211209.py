import numpy as np #Used for array operations
import matplotlib.pyplot as plt #Used to plot scatter plots of waveform data
import pandas as pd #Used to read csv files
import seaborn as sns #Used for violin plots
import time #Used for runtime optimization purposes
import pathlib #Used for checking if a file exists or not
import os #Used for checking if a file exists or not
import sys #Used to exit the program if a known error occurs
import statistics as stats #Used to calculate stdev and mean values from nparrays
from scipy.special import erfc #Used for chauvenets criterion
import getpass #Used to find the current user's username


'''Start User Imputted Settings'''
'Runthrough name'
RunName = "Medtronic Data 20210806" #run name file 
#put all excel file folders in ''C:\Users\[user]\Documents\Waveforms\Medronic Tests 20210806'' for RunName = "Medtronic Data 20210806"
# -Current Run Names-
# "20210804 Medtronic Manuscript Tests" 
# "Medtronic Data 20210806"
# "20210804 Medtronic Manuscript Tests alt" 

'Code Settings Parameters'  
fontsize = 18; plt.rcParams.update({'font.size': int(fontsize)}) #Sets the fonts for plots 
plottype = 0 #Violin(0) or Bar Graph(1)
NoiseLevel = 0.1 #Value used if noisy data occurs. Provides a lower value for force values. Only used when curve fitting occurs
NoiseLimit = 0.1  #Value used if noisy data occurs. Provides a lower limit for force values. Only used when curve fitting occurs
Show_str = 0 #are average string counts shown
Show_n = 0 #are n = [testcount] shown
legendOn = 0 #is the legend shown for waveplots
timeshift = 75 #amount shifted on the x axis for optimal plot viewing

'Reasonable value limitaions and thresholds'
Flimit = 3 #Limit for how low the Max Force value can be and still be reasonable
Flimit2 = 20 #Upper bound limit for a reasonable peak force value
Alimit = 250 #Limit for how low the Area value can be and still be reasonable
Slimit = 0 #Limit for how low the slope value can be and still be reasonable
waveend = 1.5 #What force value is considered below the threshold for the impact duration to be considered finished 
'''End User Imputted Settings'''


'''Start Analysis Program'''
'Start of Program Runtime (For Code Optimization)'
t_i = time.time() #Varible used to note time the script started. Used for runtime optimization purposes

'Initializing Variables to Begin Program Runthrough'
kN = 4.44822162 #converts voltage to lbf to kN (1V = 1000lbf for our sensor)
mVtoa = 1.090 #converts milli-volts to acceleration (1V = 1000lbf for our sensor)
Maxav = 10 #What the picoscope range for acceleration was set to (Should stay the same unless another accelerometer is used)

'Link to Teams Sharepoint Directory' #not in use yet
#Teams sharepoint file link - https://mstate.sharepoint.com/sites/MSU-MedtronicUseConditionsProgramUserCreated/Shared%20Documents/Forms/AllItems.aspx?FolderCTID=0x0120008CD6088CCEE3724F9CFFA0F8BC8B1FFA&viewid=9340f526%2D43cb%2D4070%2D9e87%2D9b32769ee004&id=%2Fsites%2FMSU%2DMedtronicUseConditionsProgramUserCreated%2FShared%20Documents%2FGeneral%2F4Python%20Processed%20Data

'Current User`s Username'
User = getpass.getuser() #Returns the current user's username

'ini File Exists Check'
Directory = "C://Users//" + User + "//Documents//Waveforms//Dataformater//"
Errmsg = ""
filecheck = np.array(['Settings',RunName,'database']) #Test file folder
if os.path.exists(pathlib.Path(Directory+filecheck[1]+'.ini')) != True: Errmsg = Errmsg + "Err - " + RunName + " File is missing " #can the run file be located
if os.path.exists(pathlib.Path(Directory+filecheck[2]+'.ini')) != True: Errmsg = Errmsg + "Err - database File is missing " #can the database file be located
if Errmsg != "": sys.exit(Errmsg) #Error designates that an important file is missing

'Locating/Creating Folder to store Graphs and Strikes'
WaveFormsfolder = pathlib.Path("C://Users//" + User + "//Documents//Waveforms")
if os.path.exists(WaveFormsfolder) != True: os.mkdir(WaveFormsfolder) #if folder does not exist create it
Codefolder = pathlib.Path("C://Users//" + User + "//Documents//Waveforms//Dataformater")
if os.path.exists(Codefolder) != True: os.mkdir(Codefolder) #if folder does not exist create it
Datafolder = pathlib.Path("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data")
if os.path.exists(Datafolder) != True: os.mkdir(Datafolder) #if folder does not exist create it
Graphfolder = pathlib.Path("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName)
if os.path.exists(Graphfolder) != True: os.mkdir(Graphfolder) #if folder does not exist create it

'Funtions Used'
#Plots analysis charts
def chart(fignum,xarray,yarray,tname,yname,xname,pltt):
    plt.figure(int(fignum),figsize=(18,10)); plt.title(tname)
    plt.rc('axes', titlesize=int(fontsize)+12) #fontsize of the title
    plt.rc('axes', labelsize=int(fontsize)+4) #fontsize of the x and y labels
    if pltt == 0: sns.violinplot(x=xarray, y=yarray); plt.ylabel(yname); plt.xlabel(xname); plt.tight_layout(); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + tname + " violin graph.png")
    if pltt == 1: plt.boxplot(x=xarray, y=yarray); plt.ylabel(yname); plt.xlabel(xname); plt.tight_layout(); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + tname + " box graph.png") 
    plt.show() 
    
#Plots waveform scatterpolts    
Count = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100])
def waveplot(fignum,strnum,testnum,incnum,tname,xname,yname,pltt,legend): 
    plt.figure(int(fignum),figsize=(18,10)); plt.grid(True); plt.tight_layout(); plt.title(tname); plt.ylabel(yname); plt.xlabel(xname)
    plt.rc('axes', titlesize=int(fontsize)+12) #fontsize of the title
    plt.rc('axes', labelsize=int(fontsize)+4) #fontsize of the x and y labels
    if pltt == 0: plt.plot(t_rec[testnum,1:incnum,strnum], F_rec[testnum,1:incnum,strnum], marker='.', label = "Strike " + str(strnum + 1))   
    if pltt == 1: plt.plot(t_rec[testnum,1:incnum,strnum], a_rec[testnum,1:incnum,strnum], marker='.', label = "Strike " + str(strnum + 1))   
    if pltt == 2: plt.plot(Count[1:incnum+1], Fmax[testnum,0:incnum], marker='.')
    if legend == 1: plt.legend(loc = "upper left") 

#Creates an array only containing information to be used in statistical analysis
def statarray(snum,stattype):
    selected = np.empty(0)
    if stattype == 0:
        for j in range(0, snum): selected = np.append(selected, Area[test][j]) 
    if stattype == 1:       
        for j in range(0, snum): selected = np.append(selected, Fmax[test][j])
    if stattype == 2:           
        for j in range(0, snum): selected = np.append(selected, IniSlope[test][j]) 
    if stattype == 3:            
        for j in range(0, snum): selected = np.append(selected, Wavelength[test][j]) 
    return selected    

#Formats a group name for the analysis charts
def NamingFormat(gnum, snum, tnum, count, incstr, incn):
    gname = GroupAbrv[gnum] 
    if incstr == 1:
        if ManuStrike == False: Avgstr = str(round(snum/tnum,1))
        else: Avgstr = str(count)
        gname += " " + Avgstr + " str"
    if incn == 1: gname += " n = " + str(tnum)
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
            word = ""; ch = file.read(1)
            while ch != "~":
                word = word + ch; ch = file.read(1)        
            if word == strmatch: 
                ch = file.read(1)
                if CF == 1: file.close() 
                return 1
    if CF == 1: file.close() 
    return 0

#Reads ini files and returns an array of values or a single value
def filereader(CF):
    ch = file.read(1) #ch is use to read characters from a txt file. Strings {$ , ~ : *} are used to activate certain commands 
    devarray = np.array([""]) #Array made up of the group names analysed in this run through
    while ch != "$":
        word = "" #word collects ch charcters into words from the txt file read
        while ch != ",":
            word = word + ch; ch = file.read(1)
        if word != "":
            if devarray[0] == "": devarray = word 
            else: devarray = np.append(devarray, word)   
        ch = file.read(1); ch = file.read(1)
    ch = file.read(1)
    if CF == 1: file.close()  
    return devarray

'Reading Run File'
file = open('C:\\Users\\' + User + '\\Documents\\Waveforms\\Dataformater\\' + RunName + '.ini', 'r') #File With Groupnames used in this run
GroupName = filereader(0) #Collecting Group Names
GroupAbrv = filereader(1) #Collecting Group Name Abreviations
            
'Obtianing a count of groups for this runthough'   
if isinstance(GroupName, np.ndarray) == False: groupct = 1
else: groupct = len(GroupName)

'Violin Chart Varible Definitions and Parameters'
if groupct > 1: GroupGrouping = True #Show graphs comparing groups
else: GroupGrouping = True 
ImpulseData = np.empty([0,4]) #collects the maxforce, area, initial slope, and impact duration between tests
TestNameData = np.empty(0) #collects the testnames to be compared. Uses Abrevation names
ImpulseData2 = np.empty([0,4]) #collects the maxforce, area, initial slope, and impact duration between groups
GroupNameData = np.empty(0) #collects the groupnames to be compared
Recdata = np.empty([0,5])
'Runthrough Loop'
Group = 0 #initiallizing Group count at 0
while Group < groupct: #Main loop that process wave form data. Repeted until all groups have been analyzed
    
    'Reading group matrix database file'
    file = open('C:\\Users\\' + User + '\\Documents\\Waveforms\\Dataformater\\database.ini', 'r')  #File With Group parameters used in this run
    if groupct > 1: Findstr = GroupName[Group]
    else: Findstr = GroupName
    if filefindmatch(Findstr, 0) == 1: GroupParameter = filereader(1) #Finding Group Parameters
    else: sys.exit('Group Parameters not found for: ' + Findstr)   

    'Obtaining Group Parameters'   
    Folder = GroupParameter[0] #Test Group Name      
    ManuCount = float(GroupParameter[1]) #Manualy imputted average strike count for the group
    cond1 = bool(int(GroupParameter[2])) #Should the trigger point be closer torward the peak (usually makes the plot look better) 
    TestName = GroupParameter[3:int(3+(len(GroupParameter)-3)/2):] #Test filenames
    Abrv = GroupParameter[int(3+(len(GroupParameter)-3)/2)::] #Test filename abreviations
        
    'Is the strike count average for the group manually imputted'
    if ManuCount != 0: ManuStrike = 1 #yes
    else: ManuStrike = 1 #no
        
    'Obtianing a count of tests for this group'      
    if isinstance(TestName, np.ndarray) == False: testct = 1
    else: testct = len(TestName)            
    
    'Analyze CSV Files'
    test = 0; Striketotal = 0
    while test < testct:
        print('flag - Start Test:' + TestName[test]) 
        filelocation = "C:\\Users\\" + User + "\\Documents\\Waveforms\\" + Folder + "\\" + TestName[test] + "\\" + TestName[test] + "_1.csv" #Test file folder
        path = pathlib.Path(filelocation); StrikeExists = os.path.exists(path)    
        if StrikeExists == False:
            filelocation = "C:\\Users\\" + User + "\\Documents\\Waveforms\\" + Folder + "\\" + TestName[test] + "\\" + TestName[test] + "_01.csv" #Test file folder    
            path = pathlib.Path(filelocation); StrikeExists = os.path.exists(path)         
        strike = 0 #start at zero for array indexing simplicity 
        if StrikeExists == False: print("Error strike file {" + str(TestName[test]) + "_1.csv or " + str(TestName[test]) + "_01.csv}  missing")
        while StrikeExists == True:
            print('flag - Start Strike:' + str(strike+1))
            
            'Setting up csv file'
            datacsv = pd.read_csv(filelocation, delimiter=',', dtype = str); datacsv = datacsv.to_numpy(dtype = str)
            Factor = int(len(datacsv)/20000)
            if len(datacsv)/20000 >= 2: data = datacsv[0::Factor,:]
            else: data = datacsv
            
            'Is accelerometer data present'
            if len(data[0]) != 3: accelerometer = 0 #no
            else: accelerometer = 1                 #yes
            
            'Setting up time units'
            if test == 0 and strike == 0:
                TimeUnit = data[0][0]                  #Time units recorded
                if TimeUnit == '(s)': mult=1E6         #(300 micro seconds for s time scale)
                elif TimeUnit == '(ms)': mult=1E3      #(300 micro seconds for ms time scale)
                elif TimeUnit == '(us)': mult=1        #(300 micro seconds for us time scale)
              
            'Data Collection/Formating/Operations Loops'
            if strike == 0:
                timerange = 300 #capture in microseconds
                del_t = (float(data[6][0]) - float(data[5][0]))*mult #the time distance between each data point
                inc = int(timerange/(del_t*2)) #Range recored before and after the data point trigger event occurs in intergers
                ratio = inc/12000 #Ratio manipulates number of points used in certain loops depending on the sample density considered                
                    
                'induvidual strike recording arrays'
                if test == 0:
                    'Defining Variables'
                    shift = int(timeshift*(inc*2/timerange)) #amount of datapoints shifted for optimal plot viewing
                    Strikesrej = 0 #number of strikes that have been thrown out
                    striketriggered = False #has a strike occured yet
                    
                    'Defining Array Variables'
                    strikecount = np.zeros([testct]) #total number of strikes analyzed by the python script from each test
                    t_rec = np.empty([testct,inc*2+2,100]) #time stamp array
                    F_rec = np.empty([testct,inc*2+2,100]) #impact force array
                    a_rec = np.empty([testct,inc*2+2,100]) #accelerometer value array
                    Area = np.zeros([testct,100]) #uses a right reamond sum to find an approximate area under the curve
                    Fmax = np.zeros([testct,100]) #Finds the max force that occurs on each impact wave 
                    IniSlope = np.zeros([testct,100]) #Finds the approximate initial slope that occurs on each impact wave
                    Wavelength = np.zeros([testct,100]) #Finds the wavelength of each impact wave
                    Rejbool = np.zeros([100]) #Determines what strikes to be shown on waveform graph
                    
                    'Statistics'
                    Areamean = np.zeros([testct]); MaxFmean = np.zeros([testct]); IniSlopemean = np.zeros([testct]); Wavelengthmean = np.zeros([testct])       #mean value arrays
                    AreaSTDev = np.zeros([testct]); MaxFSTDev = np.zeros([testct]); IniSlopeSTDev = np.zeros([testct]); WavelengthSTDev = np.zeros([testct])   #standard deviation value arrays
           
            cond1 = False #Can be the cause of certain errors if set to true
            
            'Strike analysis loop'        
            x=4 #starting point for data reading in data array
            endloop = False #used to determine is a strike has been read from the current csv file yet and ends the strike analysis loop if so
            FittingOn = False #assume curve-fitting is not need initially
            data[x-1, 1] = 0 #avoiding error in slope check in the case that data[x-1, 1] = '∞'
            while x < int(0.75*len(data))-1: #check from 0% to 0.75% of data for trigger event. If strike doesnt occur in this range, assume data is not valid
                if data[x, 1] == 'âˆž' or data[x, 1] == '∞': data[x, 1]  = '5'; FittingOn = True #Replacing Voltage Overlaod Symbols with 5V (max voltage value)    
                if data[x+300, 1] == 'âˆž' or data[x+300, 1] == '∞': data[x+300, 1]  = '5'; FittingOn = True #Replacing Voltage Overlaod Symbols with 5V (max voltage value)
                if data[x, 1] == '-∞' or float(data[x, 1]) < 0: data[x, 1] = '0'; FittingOn = True #Replacing Voltage Underload Symbols with 0V (min voltage value)
                if data[x+300, 1] == '-∞' or float(data[x+300, 1]) < 0: data[x+300, 1] = '0'; FittingOn = True #Replacing Voltage Underload Symbols with 0V (min voltage value)
                slopecheck = (float(data[x, 1]) - float(data[x-1,1])) / (float(data[x,0]) - float(data[x-1,0])) #see if the current slope is reasonable or not
                pass1 = True #Used when special conditions are considered in whether the strike data triggers analysis or not
                if cond1 == True:
                    if float(data[x+300, 1]) < 0.25*float(data[x, 1]): pass1 = False
                if x >= inc + 10 and pass1 == True and all(data[x-250:x+250, 1] != '5') and all(data[x-250:x+250,1] != 'âˆž') and all(data[x-250:x+250, 1] != '∞') and float(data[x, 1]) >= Flimit/kN and slopecheck >= Slimit: #Data Collection Triggered by 40mV signal occuring  
                    striketriggered = True
                    for n in range(x-inc+shift, x+inc+shift):
                        
                        'Collecting Data From File'
                        if n <= x+inc+shift and n < int(len(data)) - 1:
                            Array_ct = int(n-(x-inc+shift))
                            t_rec[test][Array_ct][strike] = (float(data[n, 0]) - float(data[int(x-inc+shift), 0]))*mult #t[Array_ct+offset] #Formating Time Units   
                            if data[n, 1] == 'âˆž' or data[n, 1] == '∞': #Replacing Voltage Overlaod Symbols with 5V (max voltage value)
                                data[n, 1]  = '5'
                            F_rec[test][Array_ct][strike] = float(data[n, 1]) * kN #Formating Force Units
                            if accelerometer == True: 
                                if data[n, 2] == 'âˆž' or data[n, 2] == '∞': #Replacing Voltage Overlaod Symbols with the max voltage value
                                    data[n, 2] = str(Maxav)
                                if data[n, 2] == '-∞': #Replacing Voltage Underlaod Symbols with the min voltage value
                                    data[n, 2] = str(-1*Maxav)
                                a_rec[test][Array_ct][strike] = float(data[n, 2]) / mVtoa #Formating "Useless" Data
                        else:
                            Array_ct = int(n-(x-inc+shift))  
                            t_rec[test][Array_ct][strike] = t_rec[test][Array_ct-1][strike] + t_rec[test][Array_ct-2][strike] - t_rec[test][Array_ct-3][strike]  
                            F_rec[test][Array_ct][strike] = F_rec[test][Array_ct-1][strike]
                            if accelerometer == True: 
                                a_rec[test][Array_ct][strike] = a_rec[test][Array_ct-1][strike] #Formating "Useless" Data     
                    
                    'Curve fitting(force)' 
                    Residue = 1000
                    delta = (t_rec[test][12][strike] - t_rec[test][11][strike])*mult
                    if FittingOn == True:
                        startinterp = False        
                        for j in range(0, 2*inc-1):
                            if startinterp == False:
                                if (abs(F_rec[test][j+1][strike] - F_rec[test][j][strike]))/(delta) >= 10:
                                    startinterp = True; st = j       
                            if startinterp == True:
                                if (abs(F_rec[test][j+1][strike] - F_rec[test][j][strike]))/(delta) <= 2 and F_rec[test][j][strike] < 4 * kN and F_rec[test][j][strike] >= 0:
                                    startinterp = False; en = j
                                    interp_slope = (F_rec[test][en+1][strike] - F_rec[test][st-Residue][strike]) / (t_rec[test][en][strike] - t_rec[test][st-Residue][strike])
                                    for k in range (st-Residue,en):    
                                        F_rec[test][k][strike] = F_rec[test][k-1][strike] + interp_slope*(delta)
                                        j=k
                                if j == 2*inc-2:
                                    startinterp = False; en = j + 1
                                    interp_slope = (0 - F_rec[test][st-Residue][strike]) / (t_rec[test][en][strike] - t_rec[test][st-Residue][strike])
                                    for k in range (st-Residue,en+1):    
                                        F_rec[test][k][strike] = F_rec[test][k-1][strike] + interp_slope*(delta)
                                        j=k
                            if F_rec[test][Array_ct][strike] <= NoiseLimit * kN: F_rec[test][Array_ct][strike] = NoiseLevel * kN #Silencing Values below the noise level threshold
                    
                    'Determineing Characteristics of Waveforms'
                    Area[test][strike] = 0; Fmax[test][strike] = 0; IniSlope[test][strike] = 0
                    inislope_points_cap = int(500*ratio)
                    waveduration = False
                    for j in range(0, 2*inc):
                        Area[test][strike] = Area[test][strike] + (F_rec[test][j][strike] * del_t) #Calculating via running sum                  
                        if Fmax[test][strike] < F_rec[test][j][strike]: 
                            Fmax[test][strike] = F_rec[test][j][strike] #Finding Max Force of the Wave
                        if j <= (inc + inislope_points_cap/2 - 1 - shift) and j > inc - inislope_points_cap/2 - shift:
                            IniSlope[test][strike] += (((F_rec[test][j][strike] - F_rec[test][j-1][strike]) / ((t_rec[test][j][strike] - t_rec[test][j-1][strike])))) / inislope_points_cap
                    for j in range(0, 2*inc): 
                        if waveduration == False and F_rec[test][j][strike] >= 3:
                            waveduration = True
                            ws = j
                        if waveduration == True and F_rec[test][j][strike] <= waveend:
                            Wavelength[test][strike] = (j-ws)*(timerange/(inc*2))
                            waveduration = "off"

                    endloop = True #signal to end the strike loop
                if endloop == True: x += len(data)-1 #each csv contains a single strike
                x += 1
            if striketriggered == False: Strikesrej += 1; print('Warning - strike:' + str(strike+1) + ' not triggered')
            strikecount[test] = strike
            striketriggered = False
            print('flag - end strike:' + str(strike+1))
            strike += 1
            
            'Does the next consecutive strike file exist'
            filename = TestName[test] + "_" + str(strike+1) + ".csv"
            filelocation = "C:\\Users\\" + User + "\\Documents\\Waveforms\\" + Folder + "\\" + TestName[test] + "\\" + filename  #Test file folder
            path = pathlib.Path(filelocation); StrikeExists = os.path.exists(path)   
            if StrikeExists == False:
                filename = TestName[test] + "_0" + str(strike+1) + ".csv"
                filelocation = "C:\\Users\\" + User + "\\Documents\\Waveforms\\" + Folder + "\\" + TestName[test] + "\\" + filename #Test file folder    
                path = pathlib.Path(filelocation); StrikeExists = os.path.exists(path)   
        
        'Forming a new array with only the valid values - 1'
        k = 0
        for j in range(0, int(strikecount[test])+1):  
            if Fmax[test][j] >= Flimit and Fmax[test][j] <= Flimit2 and Area[test][j] >= Alimit:
                Area[test][k] = Area[test][j]; Fmax[test][k] = Fmax[test][j]; IniSlope[test][k] = IniSlope[test][j]; Wavelength[test][k] = Wavelength[test][j]
                Rejbool[j] = 1
                k += 1 
            else: 
                print("Rej strike by OTR: " + str(j+1))
                Rejbool[j] = 0
        k -= 1

        'statistical analysis'
        statstrike = int(k+1) #Amount of strikes that occured throughout the test
        Areamean[test] = stats.mean(statarray(statstrike,0))
        AreaSTDev[test] = stats.stdev(statarray(statstrike,0), Areamean[test])
        MaxFmean[test] = stats.mean(statarray(statstrike,1))
        MaxFSTDev[test] = stats.stdev(statarray(statstrike,1), MaxFmean[test])
        IniSlopemean[test] = stats.mean(statarray(statstrike,2))
        IniSlopeSTDev[test] = stats.stdev(statarray(statstrike,2), IniSlopemean[test])
        Wavelengthmean[test] = stats.mean(statarray(statstrike,3))
        WavelengthSTDev[test] = stats.stdev(statarray(statstrike,3), Wavelengthmean[test])
        
        'Finding Values to be Rejected based on Chauvenets and absurdly low values'
        i = 0   
        Aselectedarray = statarray(statstrike,0); Fselectedarray = statarray(statstrike,1)
        Frej = chauvenetscrit(Fselectedarray,len(Fselectedarray),MaxFmean[test],MaxFSTDev[test])
        Arej = chauvenetscrit(Aselectedarray,len(Aselectedarray),Areamean[test],AreaSTDev[test])
        
        'Forming a new array with only the valid values - 2'
        nnn = 0
        for j in range(0, int(k+1)):  
            if Rejbool[j] == 0: nnn += 1
            if Arej[j] == False and Frej[j] == False:
                Area[test][i] = Area[test][j]; Fmax[test][i] = Fmax[test][j]; IniSlope[test][i] = IniSlope[test][j]; Wavelength[test][i] = Wavelength[test][j]
                Rejbool[j+nnn] = 1
                i += 1 
                
                'Writing data into a data frame for test graphing'
                ImpulseData = np.vstack([ImpulseData,[Area[test][j],Fmax[test][j],IniSlope[test][j],Wavelength[test][j]]]) 
                TestNameData = np.append(TestNameData,[Abrv[test]])

            else: 
                print("RejectedStrike: " + str(int(j+nnn+1)))   
                Rejbool[j+nnn] = 0
                
        if i == 0: sys.exit('Err - all strikes were rejected (setting cond1 to false will probably solve this issue)')
        
        'Displaying Updata Strike Waveforms'
        nn = 0
        while nn < i:
            if Rejbool[nn] == 1:
                'Force'
                waveplot(0,nn,test,inc*2-5,TestName[test],"Time (us)","Force (kN)",0,legendOn); plt.tight_layout(); plt.axis([0, 300, 0, 25])
                'Acceleration'
                if accelerometer == True: waveplot(1,strike,test,inc*2-5,TestName[test],"Time (us)","Acceleration (m^2/s)",1,legendOn); plt.tight_layout(); plt.axis([0, 300, -9, 7])
            nn += 1
        
        'Displaying the Waveform (Peak Force Data)'        
        waveplot(2,strike,test,i,TestName[test],"Strike Count","Force (kN)",2,0); plt.tight_layout(); plt.axis([0, i-1, 0, 20])
                
        'Save the waveplots as png files'        
        plt.figure(0); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + TestName[test] + "_impactforce.png")
        if accelerometer == True: plt.figure(1); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + TestName[test] + "_acceleration.png")              
        plt.figure(2); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + TestName[test] + "_peakforceplot.png")
        plt.close("all")
        
        'updating statistical analysis'    
        Areamean[test] = stats.mean(statarray(i,0)); AreaSTDev[test] = stats.stdev(statarray(i,0), Areamean[test])
        MaxFmean[test] = stats.mean(statarray(i,1)); MaxFSTDev[test] = stats.stdev(statarray(i,1), MaxFmean[test])
        IniSlopemean[test] = stats.mean(statarray(i,2)); IniSlopeSTDev[test] = stats.stdev(statarray(i,2), IniSlopemean[test])
        Wavelengthmean[test] = stats.mean(statarray(i,3)); WavelengthSTDev[test] = stats.stdev(statarray(i,3), Wavelengthmean[test])
         
        Striketotal += strikecount[test] + 1; test += 1 #add strike count to running sum for avg str calculation / Next test
    
    'Writing data into a data frame for group graphing'
    if GroupGrouping == True: 
        for j in range(0,int(testct)):
            ImpulseData2 = np.vstack([ImpulseData2,[Areamean[j],MaxFmean[j],IniSlopemean[j],Wavelengthmean[j]]])               
            NameString = NamingFormat(Group,Striketotal,testct,ManuCount,Show_str,Show_n); GroupNameData = np.append(GroupNameData,NameString)
            Recdata = np.vstack([Recdata,[NameString,Areamean[j],MaxFmean[j],IniSlopemean[j],Wavelengthmean[j]]]) 
            
    'Plotting statistical analysis charts by test'    
    DataCOL = np.array(["Area", "Force", "Slope","Length"]); NameCOL = np.array(["Test"])       
    ImpulseDataframe = pd.DataFrame(ImpulseData, columns = DataCOL); TestNameFrame = pd.DataFrame(TestNameData, columns = NameCOL)
    chart(3,TestNameFrame.Test,ImpulseDataframe.Area,"Area Under the Impulse Curve "+GroupName[Group],"Area (kN*us)","Test Name",plottype)
    chart(4,TestNameFrame.Test,ImpulseDataframe.Force,"Peak Force "+GroupName[Group],"Force (kN)","Test Name",plottype)
    chart(5,TestNameFrame.Test,ImpulseDataframe.Slope,"Initial Slope of the Wave "+GroupName[Group],"Slope (kN/us)","Test Name",plottype)
    chart(6,TestNameFrame.Test,ImpulseDataframe.Length,"Duration of the Impact Event "+GroupName[Group],"Duration (us)","Test Name",plottype)
    TestNameData = np.empty(0)
    ImpulseData = np.empty([0,4])
    plt.close("all")
    
    Group += 1 #Next group
    
'Plotting statistical analysis violin charts by group'  
if GroupGrouping == True:        
    DataCOL2 = np.array(["Area", "Force", "Slope","Length"]); NameCOL2 = np.array(["Group"])       
    ImpulseDataframe2 = pd.DataFrame(ImpulseData2, columns = DataCOL2); GroupNameFrame = pd.DataFrame(GroupNameData, columns = NameCOL2)      
    chart(3,GroupNameFrame.Group,ImpulseDataframe2.Area,"Area Under the Impulse Curve","Area (kN*us)","Group",plottype)
    chart(4,GroupNameFrame.Group,ImpulseDataframe2.Force,"Peak Force","Force (kN)","Group",plottype)
    chart(5,GroupNameFrame.Group,ImpulseDataframe2.Slope,"Initial Slope of the Wave","Slope (kN/us)","Group",plottype)
    chart(6,GroupNameFrame.Group,ImpulseDataframe2.Length,"Duration of the Impact Event","Duration (us)","Group",plottype)
    plt.close("all")
    
'Writing Average Test Parameter Values to csv'    
if GroupGrouping == True:    
    filename = "C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//Recorded Data.csv"
    RecCOL = np.array(["Implant","Area", "Force", "Slope","Length"])
    #Recdata = np.vstack((GroupNameData,ImpulseData2))      
    RecDataframe = pd.DataFrame(Recdata, columns = RecCOL)
    RecDataframe.to_csv(filename, encoding='utf-8')

'Display Program Runtime'
t_f = time.time() #End of Program Runtime (For Code Optimization)       
print("run time: " + str(round((t_f - t_i)/60)) + ":" + str(round((t_f - t_i)%60))) #Converting ss format to mm:ss format      
            