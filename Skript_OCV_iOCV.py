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
            if dataAh[1]<0:
                UAh = dU/dAh
            else:
                UAh = -dU/dAh
                
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
                if dataAh[1]<0:
                    UAh = dAh/dU
                else:
                    UAh = -dAh/dU
 
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
# initpath=r"C:\Users\maxim\Nextcloud\Vorlagen\Shared\Austausch_Max\Projekt_Entropie\OCV_daten\AM23NMC067"
initpath=r"C:\Users\Dominik\tubCloud\Studis\Austausch_Max\Projekt_Entropie\OCV_daten\AM23NMC065"
dateinamen = os.listdir(initpath)
dateien=textdateien = [datei for datei in dateinamen if datei.endswith('.txt')]
data=[]
dict_test={}
for datei in dateien:
        data_raw=load(initpath,datei,skip=0,separation=" ")
        data_temp=np.delete(data_raw,[1,2,3,5,10,12,13],1)
        data.append(data_temp)
        dict_test[datei]=data_temp
    
# ch=filtplot(dict_test['AM23NMC065_Can_C20_test_35gradC.txt'],9,9,plot=0,col=2)
# dis=filtplot(dict_test['AM23NMC065_Can_C20_test_35gradC.txt'],6,6,plot=0,col=2)
#%%
"""Daten erstellen"""
testnamen_OCV=[name for name in dateinamen if "iOCV" not in name]
testnamen_iOCV = [name for name in dateinamen if "iOCV" in name]
lab=["30°C","35°C","40°C","35°C iOCV"]
OCV={"ch":{},"dis":{}}
DVA={"ch":{},"dis":{}}
ICA={"ch":{},"dis":{}}
CAP={"ch":{},"dis":{}}
DeltaOCV={"ch":{},"dis":{}}   

Wirkungsgrad={}   

for t in testnamen_OCV:
    OCV["ch"][t]=filtplot(dict_test[t],9,9,plot=0,col=2)
    OCV["dis"][t]=filtplot(dict_test[t],6,6,plot=0,col=2)
    
    DVA["ch"][t]=dva(OCV["ch"][t][:,0],OCV["ch"][t][:,2],OCV["ch"][t][:,4])
    DVA["dis"][t]=dva(OCV["dis"][t][:,0],OCV["dis"][t][:,2],OCV["dis"][t][:,4])
    
    ICA["ch"][t]=ica(OCV["ch"][t][:,0],OCV["ch"][t][:,2],OCV["ch"][t][:,4],step=5)
    ICA["dis"][t]=ica(OCV["dis"][t][:,0],OCV["dis"][t][:,2],OCV["dis"][t][:,4],step=5)
    
    DeltaOCV["ch"][t]=np.array([OCV["ch"][t][:,2],OCV["ch"][t][:,4]]).T
    DeltaOCV["dis"][t]=np.array([OCV["dis"][t][:,2],OCV["dis"][t][:,4]]).T

    
    Wirkungsgrad[t]=abs(OCV["dis"][t][-1,5]/OCV["ch"][t][-1,5])
    CAP["ch"][t]=OCV["ch"][t][-1,5]
    CAP["dis"][t]=OCV["dis"][t][-1,5]
    
for t in testnamen_iOCV:
    tempdis=filtplot(dict_test[t],28,29,plot=0,col=2)
    tempdis=ocv(tempdis,29)
    tempdis = np.vstack((tempdis[1:], tempdis[0]))
    OCV["dis"][t]=tempdis
    
    tempch=filtplot(dict_test[t],38,39,plot=0,col=2)
    tempch=ocv(tempch,39)
    tempch = np.vstack((tempch[1:], tempch[0]))
    OCV["ch"][t]=tempch
    
    DVA["ch"][t]=dva(OCV["ch"][t][:,0],OCV["ch"][t][:,2],OCV["ch"][t][:,4])
    DVA["dis"][t]=dva(OCV["dis"][t][:,0],OCV["dis"][t][:,2],OCV["dis"][t][:,4])
    
    ICA["ch"][t]=ica(OCV["ch"][t][:,0],OCV["ch"][t][:,2],OCV["ch"][t][:,4],step=1)
    ICA["dis"][t]=ica(OCV["dis"][t][:,0],OCV["dis"][t][:,2],OCV["dis"][t][:,4],step=1)
    
    Wirkungsgrad[t]=abs(min(OCV["dis"][t][:,4])/min(OCV["ch"][t][:,4]))
    CAP["ch"][t]=abs(min(OCV["ch"][t][:,4]))
    CAP["dis"][t]=min(OCV["dis"][t][:,4])
#%%

#%%
"""Daten plotten"""

colors=[ '#22a15c','#00a6b3','#cc0000', '#ff8000']
c=0
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
# fig, axes = plt.subplots(3, 1, figsize=(8, 6))

# Plot 1: OCV Daten
c = 0
for t in testnamen_OCV:
    ch = OCV["ch"][t]
    dis = OCV["dis"][t]
    dQ_ch = ch[:, 4] - min(ch[:, 4])
    dQ_dis = np.flip(dis[:, 4] - min(dis[:, 4]))
    
    axes[0].plot(dQ_ch, ch[:, 2], linestyle='--', color=colors[c])
    axes[0].plot(dQ_dis, dis[:, 2], color=colors[c], label=lab[c])
    c += 1
    
for t in testnamen_iOCV:
    ch = OCV["ch"][t]
    dis = OCV["dis"][t]
    dQ_ch = ch[:, 4] - min(ch[:, 4])
    dQ_dis = np.flip(dis[:, 4] - min(dis[:, 4]))
    
    axes[0].plot(dQ_ch, ch[:, 2], linestyle='--', color=colors[c])
    axes[0].plot(dQ_dis, dis[:, 2], color=colors[c], label=lab[c])
    c += 1
        
axes[0].set_ylabel('OCV(V)')
axes[0].set_xlabel('dQ(Ah)')
axes[0].grid()
axes[0].legend()

# Plot 2: DVA Daten
c = 0
for t in testnamen_OCV:
    dva_ch = DVA["ch"][t]
    dva_dis = DVA["dis"][t]
    dvaCH_filt = savgol_filter(dva_ch[:, 1], 101, 1)
    dvaDIS_filt = savgol_filter(dva_dis[:, 1], 101, 1)
    
    axes[1].plot(dva_ch[:, 2], dvaCH_filt, linestyle='--', color=colors[c])
    axes[1].plot(dva_dis[:, 2], dvaDIS_filt, color=colors[c], label=lab[c])
    c += 1
    
for t in testnamen_iOCV:
    axes[1].plot(DVA["ch"][t][:, 2], DVA["ch"][t][:,1], linestyle='--', color=colors[c])
    axes[1].plot(DVA["dis"][t][:, 2], -DVA["dis"][t][:,1], color=colors[c], label=lab[c])
    c+=1

axes[1].set_ylabel('dV/dQ(V/Ah)')
axes[1].set_xlabel('dQ(Ah)')
axes[1].set_ylim(-0.5, 0.5)
axes[1].grid()
axes[1].legend()

# Plot 3: ICA Daten
c = 0
for t in testnamen_OCV:
    ica_ch = ICA["ch"][t]
    ica_dis = ICA["dis"][t]
    icaCH_filt = savgol_filter(ica_ch[:, 1], 101, 1)
    icaDIS_filt = savgol_filter(ica_dis[:, 1], 101, 1)
    
    axes[2].plot(ica_ch[:, 2], icaCH_filt, linestyle='--', color=colors[c])
    axes[2].plot(ica_dis[:, 2], icaDIS_filt, color=colors[c], label=lab[c])
    c += 1
    
for t in testnamen_iOCV:
    axes[2].plot(ICA["ch"][t][:, 2], ICA["ch"][t][:,1], linestyle='--', color=colors[c])
    axes[2].plot(ICA["dis"][t][:, 2], -ICA["dis"][t][:,1], color=colors[c], label=lab[c])
    c += 1
    
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
"""Delta OCV interpolieren"""

for key, sub_dict in DeltaOCV.items():
    # Bestimme die Länge des kürzesten Arrays im jeweiligen Unter-Dictionary
    min_length = min(arr.shape[0] for arr in sub_dict.values())
    
    # Neue x-Werte für die kürzeste Länge definieren
    x_new = np.linspace(0, 1, 6929)
    
    for arr_key, arr in sub_dict.items():
        # Ursprüngliche x-Werte für Interpolation definieren
        x_orig = np.linspace(0, 1, arr.shape[0])

        # Für jede Spalte eine Interpolationsfunktion erzeugen und auswerten
        interpolated_cols = []
        for col in range(arr.shape[1]):
            interp_func = interp1d(x_orig, arr[:, col], kind='linear')
            interpolated_col = interp_func(x_new)
            interpolated_cols.append(interpolated_col)

        # Spalten zu einem 2D-Array kombinieren und im originalen Dictionary speichern
        DeltaOCV[key][arr_key] = np.column_stack(interpolated_cols)
for i in DeltaOCV:
    for t in DeltaOCV[i]:
        DeltaOCV[i][t][:,1]-=min(DeltaOCV[i][t][:,1])  
        DeltaOCV[i][t][:,1]/=max( DeltaOCV[i][t][:,1])
#%%
# dU_dTNMC37 = np.array([0.000231375, 0.000129761, 7.94335E-05, 7.64809E-05, 3.43162E-05, 0.000194703, 0.000183935, 8.29E-05, -6.58E-05, -0.00015198, -0.001106566])*10
# SOC = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]

"""Delta OCV Plotten"""
ch_ar = DeltaOCV["ch"][testnamen_OCV[0]][:,0]-DeltaOCV["ch"][testnamen_OCV[1]][:,0]
dis_ar = np.flip(DeltaOCV["dis"][testnamen_OCV[0]][:,0]-DeltaOCV["dis"][testnamen_OCV[1]][:,0])

plt.plot(DeltaOCV["ch"][testnamen_OCV[0]][:,1]*100,DeltaOCV["ch"][testnamen_OCV[0]][:,0]-DeltaOCV["ch"][testnamen_OCV[1]][:,0],color=colors[2],label="charge")        
plt.plot(DeltaOCV["dis"][testnamen_OCV[0]][:,1]*100,DeltaOCV["dis"][testnamen_OCV[0]][:,0]-DeltaOCV["dis"][testnamen_OCV[1]][:,0],color=colors[1],label="discharge")        
plt.plot(DeltaOCV["ch"][testnamen_OCV[0]][:,1]*100, np.mean([ch_ar,dis_ar], axis=0),color=colors[0],label="mean")        

# plt.plot(SOC, dU_dTNMC37, '*', label="potentio")
plt.ylabel("dOCV(V)")
plt.xlabel("SOC(%)")
plt.ylim([-0.02,0.02])
plt.legend()
plt.grid()
plt.show()

#%%
# Temp=filtplot(dict_test['AM23NMC067_Can_iOCV_test_35gradC.txt'],41,41,plot=0,col=4)
# test=ocv(Temp,39)
#%%

# for t in testnamen:
#     ch=OCV["ch"][t]
#     dis=OCV["dis"][t]
#     dQ_ch=ch[:,4]-min(ch[:,4])
#     dQ_dis=np.flip(dis[:,4]-min(dis[:,4]))
#     plt.plot(dQ_ch,ch[:,2],linestyle='--',color=colors[c])
#     plt.plot(dQ_dis,dis[:,2],color=colors[c],label=lab[c])
#     c+=1

# plt.ylabel('OCV(V)')
# plt.xlabel('dQ(Ah)')
# plt.grid()
# plt.legend()
# plt.show()

# c=0
# for t in testnamen:
#     dva_ch=DVA["ch"][t]
#     dva_dis=DVA["dis"][t]
#     dvaCH_filt=savgol_filter(dva_ch[:,1], 101, 1)
#     dvaDIS_filt=savgol_filter(dva_dis[:,1],101,1)
    
#     plt.plot(dva_ch[:,2],dvaCH_filt,linestyle='--',color=colors[c])
#     plt.plot(dva_dis[:,2],dvaDIS_filt,color=colors[c],label=lab[c])
#     c+=1

# plt.ylabel('dV/dQ(V/Ah)')
# plt.xlabel('Q(Ah)')
# plt.ylim(-0.5,0.5)
# plt.grid()
# plt.legend()
# plt.show()

# c=0
# for t in testnamen:
#     ica_ch=ICA["ch"][t]
#     ica_dis=ICA["dis"][t]
#     icaCH_filt=savgol_filter(ica_ch[:,1], 101, 1)
#     icaDIS_filt=savgol_filter(ica_dis[:,1],101,1)
    
#     plt.plot(ica_ch[:,2],icaCH_filt,linestyle='--',color=colors[c])
#     plt.plot(ica_dis[:,2],icaDIS_filt,color=colors[c],label=lab[c])
#     c+=1
    
# plt.ylabel('dQ/dV(Ah/V)')
# plt.xlabel('dQ(Ah)')
# plt.xlim(3)
# plt.grid()
# plt.legend()
# plt.show()
#%%

