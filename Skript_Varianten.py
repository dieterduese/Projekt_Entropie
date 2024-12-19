# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mst
from cycler import cycler
from scipy.signal import savgol_filter
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import fmin, minimize
import glob
import matplotlib.cm as cm





"""Funktionen"""
def load (initpath,datei,skip,separation):
    
    initpath= initpath +"/" + datei
    with open(initpath, 'r') as f:
        temp_df = np.array(pd.read_csv(f, comment='~', skiprows=skip, sep=separation,
                           encoding='latin1', header=None, on_bad_lines='skip',
                           decimal='.'))
        return temp_df
    

def filtplot(array,line1,line2,plot,col):
    temp=array[array[:,1]>=line1]
    temp=temp[temp[:,1]<=line2]
    if plot: plt.plot(temp[:,0]-temp[0,0],temp[:,col])
    
    return temp


def ocv(array,line):
    temp=[]
    for i in range(len(array)):
        if array[i-1,1]==line and (array[i,1]==line+1 or array[i,1]==line-1):
            temp.append(array[i-1,:])
            
    return np.array(temp)

def dva(dataTime, dataU, dataAh):
    data = []
    for i in range(len(dataU)-1):
            dAh = dataAh[i+1] - dataAh[i]
            dU = dataU[i+1] - dataU[i]
            if dataU[0]>dataU[-1]:
                UAh = -abs(dU/dAh)
            else:
                UAh = abs(dU/dAh)
                
            if UAh!=np.inf and abs(UAh)<10000:
                data.append([dataTime[i],UAh,dataAh[i]-min(dataAh)])
    data=np.array(data)
    return data

def ica(dataTime, dataU, dataAh, step):
    data = []
    for i in range(len(dataU)-step):
            dAh = dataAh[i+step] - dataAh[i] 
            dU = dataU[i+step] - dataU[i]
            if dU!=0:
                if dataU[0]>dataU[-1]:
                    UAh = -abs(dU/dAh)
                else:
                    UAh = abs(dU/dAh)
                
            if UAh!=np.inf:
                data.append([dataTime[i],UAh,dataU[i]])
                # data.append([dataTime[i],UAh,dataAh[i]-min(dataAh)])
    data=np.array(data)
    return data

def interp(ydata, xval):
    xq = np.arange(min(xval),max(xval),0.02)    

    x = np.array(xval, dtype=float)

    vq = interp1d(x, ydata)
    yint = vq(xq)
    return np.column_stack([xq,yint])

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data[:,0]) >= stepsize)[0]+1)

#%%
"""Daten laden"""
initpath=r"C:\Users\maxim\Nextcloud\Vorlagen\Shared\Austausch_Max\Projekt_Entropie\OCV_Varianten\AMNMC065"
# initpath=r"C:\Users\Dominik\tubCloud\Studis\Austausch_Max\Projekt_Entropie\OCV_Varianten\AMNMC065"
dateinamen = os.listdir(initpath)
dateien=textdateien = [datei for datei in dateinamen if datei.endswith('.txt')]
data=[]
dict_test={}
for datei in dateien:
        data_raw=load(initpath,datei,skip=0,separation=" ")
        data_temp=np.delete(data_raw,[1,2,3,5,10,12,13],1)
        data.append(data_temp)
        dict_test[datei]=data_temp

#%%
ch=filtplot(dict_test['AM23NMC065_OCV_25deg.txt'],37,39,plot=1,col=2)
#20 26 32
#45,47  61,63
# dis=filtplot(dict_test['AM23NMC065_Can_C20_test_35gradC.txt'],6,6,plot=0,col=2)
#17 23 29
#37,39  53,55

#%%
"""Daten erstellen"""
testnamen_OCV=[name for name in dateinamen if "iOCV" not in name]
c_rate=["C/20","C/5","C/2","C/5 iOCV","C/2 iOCV"]

lines_OCV_ch=[20,26,32]
lines_OCV_dis=[17,23,29]
lines_iOCV_ch=[[46,47],[62,63]]
#hier nur Relaxationsphasen, da diese aus zwei Teilen mit unterschiedlichen 
#Abtastraten besteht und nur der letzte wert benötigt wird und die ocv Funktion so besser funktioniert
lines_iOCV_dis=[[38,39],[54,55]] 



OCV={"ch":{},"dis":{}}
DVA={"ch":{},"dis":{}}
ICA={"ch":{},"dis":{}}
CAP={"ch":{},"dis":{}}
DeltaOCV={"ch":{},"dis":{}}   

Wirkungsgrad={}   
testnamen_OCV=["AM23NMC065_OCV_45deg.txt"]
for t in testnamen_OCV:
    l=0
    OCV["ch"][t]={}
    DVA["ch"][t]={}
    ICA["ch"][t]={}
    CAP["ch"][t]={}
    DeltaOCV["ch"][t]={}
    Wirkungsgrad[t]={}
    for i in lines_OCV_ch:
        temp=filtplot(dict_test[t],i,i,plot=0,col=2)
        unique_values, unique_indices = np.unique(temp[:, 2], return_index=True)
        unique_indices.sort()
        filtered_data = temp[unique_indices]
        OCV["ch"][t][c_rate[l]]=filtered_data
        DVA["ch"][t][c_rate[l]]=dva(OCV["ch"][t][c_rate[l]][:,0],OCV["ch"][t][c_rate[l]][:,2],OCV["ch"][t][c_rate[l]][:,4])
        ICA["ch"][t][c_rate[l]]=ica(OCV["ch"][t][c_rate[l]][:,0],OCV["ch"][t][c_rate[l]][:,2],OCV["ch"][t][c_rate[l]][:,4],step=5)
        DeltaOCV["ch"][t][c_rate[l]]=np.array([OCV["ch"][t][c_rate[l]][:,2],OCV["ch"][t][c_rate[l]][:,4]]).T
        CAP["ch"][t][c_rate[l]]=OCV["ch"][t][c_rate[l]][-1,5]
        print(i)
        l+=1
    for i in lines_iOCV_ch:
        tempch=filtplot(dict_test[t],i[0],i[1],plot=0,col=2)
        tempch=ocv(tempch,i[1])
        tempch = np.vstack((tempch[1:], tempch[0]))
        OCV["ch"][t][c_rate[l]]=tempch
        DVA["ch"][t][c_rate[l]]=dva(OCV["ch"][t][c_rate[l]][:,0],OCV["ch"][t][c_rate[l]][:,2],OCV["ch"][t][c_rate[l]][:,4])
        ICA["ch"][t][c_rate[l]]=ica(OCV["ch"][t][c_rate[l]][:,0],OCV["ch"][t][c_rate[l]][:,2],OCV["ch"][t][c_rate[l]][:,4],step=5)
        CAP["ch"][t][c_rate[l]]=abs(min(OCV["ch"][t][c_rate[l]][:,4]))
        print(i)
        l+=1
    l=0
    OCV["dis"][t]={}
    DVA["dis"][t]={}
    ICA["dis"][t]={}
    CAP["dis"][t]={}
    DeltaOCV["dis"][t]={}

    for i in lines_OCV_dis:
        temp=filtplot(dict_test[t],i,i,plot=0,col=2)
        unique_values, unique_indices = np.unique(temp[:, 2], return_index=True)
        unique_indices.sort()
        filtered_data = temp[unique_indices]
        OCV["dis"][t][c_rate[l]]=filtered_data
        DVA["dis"][t][c_rate[l]]=dva(OCV["dis"][t][c_rate[l]][:,0],OCV["dis"][t][c_rate[l]][:,2],OCV["dis"][t][c_rate[l]][:,4])
        ICA["dis"][t][c_rate[l]]=ica(OCV["dis"][t][c_rate[l]][:,0],OCV["dis"][t][c_rate[l]][:,2],OCV["dis"][t][c_rate[l]][:,4],step=5)
        DeltaOCV["dis"][t][c_rate[l]]=np.array([OCV["dis"][t][c_rate[l]][:,2],OCV["dis"][t][c_rate[l]][:,4]]).T
        CAP["dis"][t][c_rate[l]]=OCV["dis"][t][c_rate[l]][-1,5]
        Wirkungsgrad[t][c_rate[l]]=abs(OCV["dis"][t][c_rate[l]][-1,5]/OCV["ch"][t][c_rate[l]][-1,5])
        print(i)
        l+=1
        
    for i in lines_iOCV_dis:
        tempdis=filtplot(dict_test[t],i[0],i[1],plot=0,col=2)
        tempdis=ocv(tempdis,i[1])
        tempdis = np.vstack((tempdis[1:], tempdis[0]))
        OCV["dis"][t][c_rate[l]]=tempdis
        DVA["dis"][t][c_rate[l]]=dva(OCV["dis"][t][c_rate[l]][:,0],OCV["dis"][t][c_rate[l]][:,2],OCV["dis"][t][c_rate[l]][:,4])
        ICA["dis"][t][c_rate[l]]=ica(OCV["dis"][t][c_rate[l]][:,0],OCV["dis"][t][c_rate[l]][:,2],OCV["dis"][t][c_rate[l]][:,4],step=1)
        CAP["dis"][t][c_rate[l]]=min(OCV["dis"][t][c_rate[l]][:,4])
        Wirkungsgrad[t][c_rate[l]]=abs(min(OCV["dis"][t][c_rate[l]][:,4])/min(OCV["ch"][t][c_rate[l]][:,4]))
        print(i)
        l+=1

    #%%
plt.plot(DVA["dis"]["AM23NMC065_OCV_25deg.txt"]["C/20"][:,2],savgol_filter(DVA["dis"]["AM23NMC065_OCV_25deg.txt"]["C/20"][:,1],1001,1),label=("C/20"))
plt.plot(DVA["dis"]["AM23NMC065_OCV_25deg.txt"]["C/5"][:,2],savgol_filter(DVA["dis"]["AM23NMC065_OCV_25deg.txt"]["C/5"][:,1],501,1),label=("C/5"))
plt.plot(DVA["dis"]["AM23NMC065_OCV_25deg.txt"]["C/2"][:,2],savgol_filter(DVA["dis"]["AM23NMC065_OCV_25deg.txt"]["C/2"][:,1],201,1),label=("C/2"))

# plt.plot(DVA["ch"]["AM23NMC065_OCV_25deg.txt"]["C/20"][:,1])
plt.ylim(-0.5,0.5)
plt.legend()
plt.show()
#%%
"""Alles plotten"""

# colors=[ '#22a15c','#00a6b3','#cc0000', '#ff8000','#9900cc']
colors=['#1f77b4', '#39a2db', '#2ca02c', '#00bfae', '#009e74', '#e60000','#ff0000', '#ff6600', '#ff9900', '#ffcc00']
c=0
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
# fig, axes = plt.subplots(3, 1, figsize=(8, 6))

# Plot 1: OCV Daten
temp=["25°C","45°C"]
a=0
savgol={"C/20":101,"C/5":91,"C/2":61}
for t in testnamen_OCV:
    for i in c_rate:
        ch = OCV["ch"][t][i]
        dis = OCV["dis"][t][i]
        dQ_ch = ch[:, 4] - min(ch[:, 4])
        dQ_dis = np.flip(dis[:, 4] - min(dis[:, 4]))
    
        axes[0].plot(dQ_ch, ch[:, 2], linestyle='--', color=colors[c])
        axes[0].plot(dQ_dis, dis[:, 2], color=colors[c],label=temp[a]+" "+i)
        
        
        dva_ch = DVA["ch"][t][i]
        dva_dis = DVA["dis"][t][i]
        if "iOCV" not in i:
            dvaCH_filt = savgol_filter(dva_ch[:, 1], savgol[i], 1)
            dvaDIS_filt = savgol_filter(dva_dis[:, 1], savgol[i], 1)
        if "iOCV" in i:
            dvaCH_filt=dva_ch[:,1]
            dvaDIS_filt=dva_dis[:,1]
        axes[1].plot(dva_ch[:, 2], dvaCH_filt, linestyle='--', color=colors[c])
        axes[1].plot(dva_dis[:, 2], dvaDIS_filt, color=colors[c], label=temp[a]+" "+i)
        
        
        ica_ch = ICA["ch"][t][i]
        ica_dis = ICA["dis"][t][i]
        if "iOCV" not in i:
            icaCH_filt = savgol_filter(ica_ch[:, 1], 101, 1)
            icaDIS_filt = savgol_filter(ica_dis[:, 1], 101, 1)
        else:
            icaCH_filt=ica_ch[:,1]
            icaDIS_filt=ica_dis[:,1]
        
        axes[2].plot(ica_ch[:, 2], icaCH_filt, linestyle='--', color=colors[c])
        axes[2].plot(ica_dis[:, 2], icaDIS_filt, color=colors[c], label=temp[a]+" "+i)
        
        c += 1
    a+=1

axes[0].set_ylabel('OCV(V)')
axes[0].set_xlabel('dQ(Ah)')
axes[0].grid()
axes[0].legend()
axes[1].set_ylabel('dV/dQ(V/Ah)')
axes[1].set_xlabel('dQ(Ah)')
axes[1].set_ylim(-0.5, 0.5)
axes[1].grid()
axes[1].legend()
axes[2].set_ylabel('dQ/dV(Ah/V)')
axes[2].set_xlabel('U(V)')
axes[2].set_xlim(left=3)
axes[2].grid()
axes[2].legend()

# Passe das Layout der Subplots an
plt.tight_layout()

# Zeige den Plot
plt.show()



   #%%
"""C-Raten plotten"""
colors=[ '#22a15c','#00a6b3','#cc0000', '#ff8000','#808080']
# fig, axes = plt.subplots(3, 1, figsize=(8, 6))

# Plot 1: OCV Daten
temp=["25°C","45°C"]
a=0
savgol={"C/20":101,"C/5":91,"C/2":61}
for t in testnamen_OCV:
    c=0
    fig, axes = plt.subplots(3, 1, figsize=(8, 15))

    for i in c_rate:
        ch = OCV["ch"][t][i]
        dis = OCV["dis"][t][i]
        dQ_ch = ch[:, 4] - min(ch[:, 4])
        dQ_dis = dis[:, 4] - min(dis[:, 4])
    
        axes[0].plot(dQ_ch, ch[:, 2], linestyle='--', color=colors[c])
        axes[0].plot(dQ_dis, dis[:, 2], color=colors[c],label=temp[a]+" "+i)
        
        
        dva_ch = DVA["ch"][t][i]
        dva_dis = DVA["dis"][t][i]
        if "iOCV" not in i:
            dvaCH_filt = savgol_filter(dva_ch[:, 1], savgol[i], 1)
            dvaDIS_filt = savgol_filter(dva_dis[:, 1], savgol[i], 1)
        if "iOCV" in i:
            dvaCH_filt=dva_ch[:,1]
            dvaDIS_filt=dva_dis[:,1]
        axes[1].plot(dva_ch[:, 2], dvaCH_filt, linestyle='--', color=colors[c])
        axes[1].plot(dva_dis[:, 2], dvaDIS_filt, color=colors[c], label=temp[a]+" "+i)
        
        
        ica_ch = ICA["ch"][t][i]
        ica_dis = ICA["dis"][t][i]
        if "iOCV" not in i:
            icaCH_filt = savgol_filter(ica_ch[:, 1], 101, 1)
            icaDIS_filt = savgol_filter(ica_dis[:, 1], 101, 1)
        else:
            icaCH_filt=ica_ch[:,1]
            icaDIS_filt=ica_dis[:,1]
        
        axes[2].plot(ica_ch[:, 2], icaCH_filt, linestyle='--', color=colors[c])
        axes[2].plot(ica_dis[:, 2], icaDIS_filt, color=colors[c], label=temp[a]+" "+i)
        
        c += 1
    axes[0].set_ylabel('OCV(V)')
    axes[0].set_xlabel('dQ(Ah)')
    axes[0].grid()
    axes[0].legend()
    axes[1].set_ylabel('dV/dQ(V/Ah)')
    axes[1].set_xlabel('dQ(Ah)')
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].grid()
    axes[1].legend()
    axes[2].set_ylabel('dQ/dV(Ah/V)')
    axes[2].set_xlabel('U(V)')
    axes[2].set_xlim(left=3)
    axes[2].grid()
    axes[2].legend()
    
    # Passe das Layout der Subplots an
    plt.tight_layout()
    
    # Zeige den Plot
    plt.show()
    a+=1



#%%
"""Temp vergleich"""
colors=[ '#00a6b3','#cc0000','#22a15c', '#ff8000','#9900cc']
c=0
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
# fig, axes = plt.subplots(3, 1, figsize=(8, 6))

# Plot 1: OCV Daten
temp=["25°C","45°C"]
a=0
savgol={"C/20":101,"C/5":91,"C/2":61}
for t in testnamen_OCV:
    i="C/20"    
    ch = OCV["ch"][t][i]
    dis = OCV["dis"][t][i]
    dQ_ch = ch[:, 4] - min(ch[:, 4])
    dQ_dis = np.flip(dis[:, 4] - min(dis[:, 4]))

    axes[0].plot(dQ_ch, ch[:, 2], linestyle='--', color=colors[c])
    axes[0].plot(dQ_dis, dis[:, 2], color=colors[c],label=temp[a]+" "+i)
    
    
    dva_ch = DVA["ch"][t][i]
    dva_dis = DVA["dis"][t][i]

    dvaCH_filt = savgol_filter(dva_ch[:, 1], savgol[i], 1)
    dvaDIS_filt = savgol_filter(dva_dis[:, 1], savgol[i], 1)
    
    axes[1].plot(dva_ch[:, 2], dvaCH_filt, linestyle='--', color=colors[c])
    axes[1].plot(dva_dis[:, 2], dvaDIS_filt, color=colors[c], label=temp[a]+" "+i)
    
    
    ica_ch = ICA["ch"][t][i]
    ica_dis = ICA["dis"][t][i]
    icaCH_filt = savgol_filter(ica_ch[:, 1], 101, 1)
    icaDIS_filt = savgol_filter(ica_dis[:, 1], 101, 1)
    axes[2].plot(ica_ch[:, 2], icaCH_filt, linestyle='--', color=colors[c])
    axes[2].plot(ica_dis[:, 2], icaDIS_filt, color=colors[c], label=temp[a]+" "+i)
    
    c += 1
    a+=1

axes[0].set_ylabel('OCV(V)')
axes[0].set_xlabel('dQ(Ah)')
axes[0].grid()
axes[0].legend()
axes[1].set_ylabel('dV/dQ(V/Ah)')
axes[1].set_xlabel('dQ(Ah)')
axes[1].set_ylim(-0.5, 0.5)
axes[1].grid()
axes[1].legend()
axes[2].set_ylabel('dQ/dV(Ah/V)')
axes[2].set_xlabel('U(V)')
axes[2].set_xlim(left=3)
axes[2].grid()
axes[2].legend()

# Passe das Layout der Subplots an
plt.tight_layout()

# Zeige den Plot
plt.show()


#%%
# """Delta OCV interpolieren"""

# for key, sub_dict in DeltaOCV.items():
#     # Bestimme die Länge des kürzesten Arrays im jeweiligen Unter-Dictionary
#     min_length = min(arr.shape[0] for arr in sub_dict.values())
    
#     # Neue x-Werte für die kürzeste Länge definieren
#     x_new = np.linspace(0, 1, 6929)
    
#     for arr_key, arr in sub_dict.items():
#         # Ursprüngliche x-Werte für Interpolation definieren
#         x_orig = np.linspace(0, 1, arr.shape[0])

#         # Für jede Spalte eine Interpolationsfunktion erzeugen und auswerten
#         interpolated_cols = []
#         for col in range(arr.shape[1]):
#             interp_func = interp1d(x_orig, arr[:, col], kind='linear')
#             interpolated_col = interp_func(x_new)
#             interpolated_cols.append(interpolated_col)

#         # Spalten zu einem 2D-Array kombinieren und im originalen Dictionary speichern
#         DeltaOCV[key][arr_key] = np.column_stack(interpolated_cols)
# for i in DeltaOCV:
#     for t in DeltaOCV[i]:
#         DeltaOCV[i][t][:,1]-=min(DeltaOCV[i][t][:,1])  
#         DeltaOCV[i][t][:,1]/=max( DeltaOCV[i][t][:,1])
# #%%
# dU_dTNMC37 = np.array([0.000231375, 0.000129761, 7.94335E-05, 7.64809E-05, 3.43162E-05, 0.000194703, 0.000183935, 8.29E-05, -6.58E-05, -0.00015198, -0.001106566])*10
# SOC = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]

# """Delta OCV Plotten"""
# ch_ar = DeltaOCV["ch"][testnamen_OCV[0]][:,0]-DeltaOCV["ch"][testnamen_OCV[1]][:,0]
# dis_ar = np.flip(DeltaOCV["dis"][testnamen_OCV[0]][:,0]-DeltaOCV["dis"][testnamen_OCV[1]][:,0])

# plt.plot(DeltaOCV["ch"][testnamen_OCV[0]][:,1]*100,DeltaOCV["ch"][testnamen_OCV[0]][:,0]-DeltaOCV["ch"][testnamen_OCV[1]][:,0],color=colors[2],label="charge")        
# plt.plot(DeltaOCV["dis"][testnamen_OCV[0]][:,1]*100,DeltaOCV["dis"][testnamen_OCV[0]][:,0]-DeltaOCV["dis"][testnamen_OCV[1]][:,0],color=colors[1],label="discharge")        
# plt.plot(DeltaOCV["ch"][testnamen_OCV[0]][:,1]*100, np.mean([ch_ar,dis_ar], axis=0),color=colors[0],label="mean")        

# plt.plot(SOC, dU_dTNMC37, '*', label="potentio")
# plt.ylabel("dOCV(V)")
# plt.xlabel("SOC(%)")
# plt.ylim([-0.02,0.02])
# plt.legend()
# plt.grid()
# plt.show()


