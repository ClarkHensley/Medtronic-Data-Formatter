import numpy as np #Used for array operations
import matplotlib.pyplot as plt #Used to plot scatter plots of waveform data
import pandas as pd #Used to read csv files
import pathlib #Used for checking if a file exists or not
import os #Used for checking if a file exists or not
import getpass #Used to find the current user's username

'''Start User Imputted Settings'''
Folder = "" #Folder within compressive data folder that data files are in (optional)
Id1 = "Implant" #First file name identifier
Id2 = "test" #Second file name identifier
#File naming convention "Id1# Id2#"

groups = 18 #how many groups(implants)
testspergroup = 8 #how many files per group(implant)
savename = "Implants (Grouped) (all data points) (refined)"
alldata = True #graph all data points or average at ecah strike
Complimit = 220
Displimit = 0
groupshow = True
showcompdisp = False
'''End User Imputted Settings'''

'Code Settings Parameters'  
fontsize = 18; plt.rcParams.update({'font.size': int(fontsize)}) #Sets the fonts for plots 
plottype = 0 #Violin(0) or Bar Graph(1)
Show_str = 0 #are average string counts shown
Show_n = 0 #are n = [testcount] shown
'''End User Imputted Settings'''

#is the legend shown for waveplots
if groupshow == True: legendOn = 1 
else: legendOn = 0

'Program Functions'
#Plots waveform scatterpolts    
def waveplot(testnum,fignum,incnum,tname,xname,yname,pltt,legend): 
    plt.figure(int(fignum),figsize=(18,10)); plt.grid(True); plt.tight_layout(); plt.title(tname); plt.ylabel(yname); plt.xlabel(xname)
    plt.rc('axes', titlesize=int(fontsize)+12) #fontsize of the title
    plt.rc('axes', labelsize=int(fontsize)+4) #fontsize of the x and y labels
    if pltt == 0: plt.plot(Count[0:incnum], Comp[0:incnum], marker='.', label = "Test " + str(testnum))     
    if pltt == 1: plt.plot(Count[0:incnum], Disp[0:incnum], marker='.', label = "Test " + str(testnum))   
    if pltt == 2: plt.plot(Comp[0:incnum], Disp[0:incnum], marker='.', label = "Test " + str(testnum))   
    if legend == 1: plt.legend(loc = "upper left") 

'Current User`s Username'
User = getpass.getuser() #Returns the current user's username

'Compression/displacement Data location'
FolderLocation = "\\Desktop\\ME Pulse Sensor Research\\Compression Force Data\\Configuration\\"

"Find a folder for plots"    
Graphfolder = pathlib.Path("C:\\Users\\" + User + FolderLocation + "\\graphs")
if os.path.exists(Graphfolder) != True: os.mkdir(Graphfolder) #if folder does not exist create it    
Graphfolder = pathlib.Path("C:\\Users\\" + User + FolderLocation + "\\graphs\\" + savename)
if os.path.exists(Graphfolder) != True: os.mkdir(Graphfolder) #if folder does not exist create it
filesaveloc = "C:\\Users\\" + User + FolderLocation + "\\graphs\\" + savename + "\\"

'Loop varible definitions'
Disp = np.empty([10000]) #Displacement array
Comp = np.empty([10000]) #Compression force array
Count = np.empty([10000])
Missing_ct = 0
File_ct = 1
test_ct = 1
add = ""
implantexists = False

"Analysis loop"
if Id2 != "": Id2 = " " + Id2 #formatting second file name identifier
while File_ct < groups + 1:

    "Pull data from file"
    #are there multiple tests?
    if testspergroup != 1: test_ctstr = str(test_ct)
    else: test_ctstr = ""
    #what is the file name to be opened?
    File = Id1 + str(File_ct) + Id2 + test_ctstr 
    #is the file located is a folder within the compressive data folder?
    if Folder != "": addin = "//"
    else: addin = ""
    #can the file be found?
    filelocation = "C:\\Users\\" + User + FolderLocation + Folder + addin + File + ".dat" #Test file folder
    path = pathlib.Path(filelocation); FileExists = os.path.exists(path)    
    if FileExists == True:
        data = pd.read_csv(filelocation, delimiter='\t', dtype = str); data = data.to_numpy(dtype = str)
        implantexists = True
        'Organize/Format data'
        x = 0
        n = 1
        i = 0
        Disp[0] = 0
        Comp[0] = data[0,1]
        Count[0] = 0
        
        'graphing data point average at each strike'
        if alldata == False:
            add = ""
            ctname = "Count (strikes)"
            while x < int(len(data)):
                if 1 < abs(float(data[x,2]) - float(data[i,2])): 
                    Disp[n] = float(data[0,2]) - float(data[x,2])   
                    Comp[n] = data[x,1]
                    Count[n] = n + 1
                    n += 1
                    i = x
                x += 1
        
        'graphing all data points'
        if alldata == True:
            ctname = "Count (percent of waveform)"
            add = "all"
            while x < int(len(data)):
                Disp[x] = float(data[0,2]) - float(data[x,2])  
                if float(data[x,1]) >= Complimit: Comp[n] = Comp[n-1]
                else: Comp[n] = data[x,1]
                Count[x] = (x + 1)/len(data)
                x += 1
                n = x
                
        'Scatter Plots'
        waveplot(test_ct,0,n-1,"Count vs Comp",ctname,"Compression (N)",0,legendOn)#; plt.tight_layout(); plt.axis([0, 300, 0, 25])
        waveplot(test_ct,1,n-1,"Count vs Disp",ctname,"Displacement (mm)",1,legendOn)#; plt.tight_layout(); plt.axis([0, 300, 0, 25])
        waveplot(test_ct,2,n-1,"Comp vs Disp","Compression (N)","Displacement (mm)",2,legendOn)#; plt.tight_layout(); plt.axis([0, 300, 0, 25])    
            
        'Save the scatter plots as png files'  
        if groupshow == False:
            plt.figure(0); plt.savefig(filesaveloc + File + "_count_vs_comp" + add + ".png")     
            plt.figure(1); plt.savefig(filesaveloc + File + "_count_vs_disp" + add + ".png")     
            if showcompdisp == True: plt.figure(2); plt.savefig(filesaveloc + File + "_comp_vs_disp" + add + ".png")          
            plt.close("all")
        
    if FileExists == True: print("File: " + File + " - Found")
    else: 
        print("File: " + File + " - Not Found")
        Missing_ct += 1
    if test_ct == testspergroup:
        if groupshow == True and implantexists == True:
            plt.figure(0); plt.savefig(filesaveloc + "Implant" + str(File_ct) + "_count_vs_comp" + add + ".png")     
            plt.figure(1); plt.savefig(filesaveloc + "Implant" + str(File_ct) + "_count_vs_disp" + add + ".png")     
            if showcompdisp == True: plt.figure(2); plt.savefig(filesaveloc + "Implant" + str(File_ct) + "_comp_vs_disp" + add + ".png")          
            plt.close("all") 
        File_ct += 1
        test_ct = 1
        implantexists = False
    else:
        test_ct += 1
        
print(str(Missing_ct) + " files are missing") 