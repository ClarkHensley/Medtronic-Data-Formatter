
        strike = 0 #start at zero for array indexing simplicity 
           
            cond1 = False #Can be the cause of certain errors if set to true
            
            'Strike analysis loop'        
            x=4 #starting point for data reading in data array
            endloop = False
            FittingOn = False #assume curve-fitting is not need initially
            data[x-1, 1] = 0 #avoiding error in slope check in the case that data[x-1, 1] = '∞'

            while x < int(0.75*len(data))-1: #check from 0% to 0.75% of data for trigger event. If strike doesnt occur in this range, assume data is not valid
                    #Replacing Voltage Underload Symbols with 0V (min voltage value)

                slopecheck = (float(data[x, 1]) - float(data[x-1,1])) / (float(data[x,0]) - float(data[x-1,0]))
                #see if the current slope is reasonable or not

                pass1 = True #Used when special conditions are considered in whether the strike data triggers analysis or not




                if cond1 == True:
                    if float(data[x+300, 1]) < 0.25*float(data[x, 1]): pass1 = False


                if x >= inc + 10 and pass1 == True and all(data[x-250:x+250, 1] != '5') and all(data[x-250:x+250,1] != 'âˆž') and all(data[x-250:x+250, 1] != '∞') and float(data[x, 1]) >= Flimit/kN and slopecheck >= Slimit: #Data Collection Triggered by 40mV signal occuring  
                    striketriggered = True
                    for n in range(x-inc+shift, x+inc+shift):
                        
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
            
            filename = TestName[test] + "_" + str(strike+1) + ".csv"
            filelocation = "C:\\Users\\" + User + "\\Documents\\Waveforms\\" + Folder + "\\" + TestName[test] + "\\" + filename
            path = pathlib.Path(filelocation); StrikeExists = os.path.exists(path)   
            if StrikeExists == False:
                filename = TestName[test] + "_0" + str(strike+1) + ".csv"
                filelocation = "C:\\Users\\" + User + "\\Documents\\Waveforms\\" + Folder + "\\" + TestName[test] + "\\" + filename
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
                
        plt.figure(0); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + TestName[test] + "_impactforce.png")
        if accelerometer == True: plt.figure(1); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + TestName[test] + "_acceleration.png")              
        plt.figure(2); plt.savefig("C://Users//" + User + "//Documents//Waveforms//Dataformater//Data//" + RunName + "//" + TestName[test] + "_peakforceplot.png")
        plt.close("all")
        
        'updating statistical analysis'    
        Areamean[test] = stats.mean(statarray(i,0)); AreaSTDev[test] = stats.stdev(statarray(i,0), Areamean[test])
        MaxFmean[test] = stats.mean(statarray(i,1)); MaxFSTDev[test] = stats.stdev(statarray(i,1), MaxFmean[test])
        IniSlopemean[test] = stats.mean(statarray(i,2)); IniSlopeSTDev[test] = stats.stdev(statarray(i,2), IniSlopemean[test])
        Wavelengthmean[test] = stats.mean(statarray(i,3)); WavelengthSTDev[test] = stats.stdev(statarray(i,3), Wavelengthmean[test])
         
        Striketotal += strikecount[test] + 1;
        test += 1 #add strike count to running sum for avg str calculation / Next test
    
    'Writing data into a data frame for group graphing'
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
