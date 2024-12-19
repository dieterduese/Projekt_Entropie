# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:55:59 2024

@author: maxim
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
import pickle






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

"""OCV Daten erstellen"""

path=r"C:\Users\maxim\Nextcloud\Vorlagen\Shared\Austausch_Max\Projekt_Entropie\OCV_Varianten"
zellnamen=os.listdir(path)
# zellnamen=["AMNMC067"]
c_rate=["C/20","C/5","C/2","C/5 iOCV","C/2 iOCV"]

OCV={}
#%%
for z in zellnamen:
    initpath=path+"/"+z
    # initpath=r"C:\Users\Dominik\tubCloud\Studis\Austausch_Max\Projekt_Entropie\OCV_Varianten\AMNMC065"
    dateinamen = os.listdir(initpath)
    dateien= [datei for datei in dateinamen if datei.endswith('.txt')]
    data=[]
    dict_test={}
    for datei in dateien:
            data_raw=load(initpath,datei,skip=0,separation=" ")
            data_temp=np.delete(data_raw,[1,2,3,5,10,12,13],1)
            data.append(data_temp)
            dict_test[datei]=data_temp
    
    
    testnamen_OCV=[name for name in dateinamen if "iOCV" not in name]
    
    lines_OCV_ch=[20,26,32]
    lines_OCV_dis=[17,23,29]
    lines_iOCV_ch=[[46,47],[62,63]]
    #hier nur Relaxationsphasen, da diese aus zwei Teilen mit unterschiedlichen 
    #Abtastraten besteht und nur der letzte wert benötigt wird und die ocv Funktion so besser funktioniert
    lines_iOCV_dis=[[38,39],[54,55]] 
    
    
    OCV[z]={"ch":{},"dis":{}}
    
    
    for test in testnamen_OCV:
        if "_5deg" in test: t="5°C"
        elif "_25deg" in test: t="25°C"
        elif "_45deg" in test: t="45°C"
        
        l=0
        OCV[z]["ch"][t]={}
       
        for i in lines_OCV_ch:
            temp=filtplot(dict_test[test],i,i,plot=0,col=2)
            unique_values, unique_indices = np.unique(temp[:, 2], return_index=True)
            unique_indices.sort()
            filtered_data = temp[unique_indices]
            OCV[z]["ch"][t][c_rate[l]]=np.asarray(filtered_data,dtype=np.float64)
    
            l+=1
            
        for i in lines_iOCV_ch:
            tempch=filtplot(dict_test[test],i[0],i[1],plot=0,col=2)
            tempch=ocv(tempch,i[1])
            tempch = np.vstack((tempch[1:], tempch[0]))
            OCV[z]["ch"][t][c_rate[l]]=np.asarray(tempch,dtype=np.float64)
    
            l+=1
            
            
        l=0
        OCV[z]["dis"][t]={}
       
        for i in lines_OCV_dis:
            temp=filtplot(dict_test[test],i,i,plot=0,col=2)
            unique_values, unique_indices = np.unique(temp[:, 2], return_index=True)
            unique_indices.sort()
            filtered_data = temp[unique_indices]
            OCV[z]["dis"][t][c_rate[l]]=np.asarray(filtered_data,dtype=np.float64)
            l+=1
            
        for i in lines_iOCV_dis:
            tempdis=filtplot(dict_test[test],i[0],i[1],plot=0,col=2)
            tempdis=ocv(tempdis,i[1])
            tempdis = np.vstack((tempdis[1:], tempdis[0]))
            OCV[z]["dis"][t][c_rate[l]]=np.asarray(tempdis,dtype=np.float64)
    
            l+=1
#%%
"""OCV Dict Speichern"""
# with open('OCV_daten.pkl', 'wb') as file:
#     pickle.dump(OCV, file)
    
"""OCV Dict laden"""
with open('OCV_daten.pkl', 'rb') as file:
    OCV = pickle.load(file)

#%%
"""Mittelwerte OCV"""

def mittelwert(data, cell_names, mode, temperature, c_rate):
    """
    Berechnet Mittelwert und Standardabweichung der Werte (Zeit, Spannung, Kapazität) für zusammengehörige Zellen.
    
    Parameter:
    - data: dict mit Daten in der Struktur {zelle: {'ch': {temperatur: {C-Rate: array}}, 'dis': {temperatur: {C-Rate: array}}}}
    - cell_names: Liste der Namen der zusammengehörigen Zellen
    - mode: 'ch' oder 'dis' für Lade- oder Entlade-Daten
    - temperature: Temperatur, für die die Daten ausgewertet werden sollen
    - c_rate: C-Rate, für die die Daten ausgewertet werden sollen
    
    Rückgabewert:
    - Ein Dictionary mit zwei Schlüsseln:
        - "mean": Ein Array der Form (min_length, 3) für die Mittelwerte von Zeit, Spannung und Kapazität
        - "std": Ein Array der Form (min_length, 3) für die Standardabweichungen von Zeit, Spannung und Kapazität
    """
    # Speichern der interpolierten Arrays
    interpolated_arrays = []
    
    # Finde die minimale Länge aller Arrays für die gegebene Temperatur und C-Rate
    min_length = min(data[cell][mode][temperature][c_rate].shape[0] for cell in cell_names)
    
    # Definiere die neuen x-Werte für die kürzeste Länge
    x_new = np.linspace(0, 1, min_length)
    
    for cell in cell_names:
        # Hole das Array der entsprechenden Zelle, Temperatur und C-Rate
        array = data[cell][mode][temperature][c_rate]
        
        # Ursprüngliche x-Werte basierend auf der Länge des Arrays definieren
        x_orig = np.linspace(0, 1, array.shape[0])

        # Für jede Spalte eine Interpolationsfunktion erstellen und interpolieren
        interpolated_cols = []
        for col in range(array.shape[1]):
            interp_func = interp1d(x_orig, array[:, col], kind='linear')
            interpolated_col = interp_func(x_new)
            interpolated_cols.append(interpolated_col)

        # Spalten zu einem 2D-Array kombinieren und hinzufügen
        interpolated_array = np.column_stack(interpolated_cols)
        interpolated_arrays.append(interpolated_array)
    
    # Stacke die interpolierten Arrays entlang der dritten Achse, um eine Form (min_length, n, num_cells) zu erhalten
    stacked_data = np.stack(interpolated_arrays, axis=-1)
    
    # Berechne Mittelwert und Standardabweichung entlang der Zellen-Achse
    mean_values = np.mean(stacked_data, axis=2)
    std_values = np.std(stacked_data, axis=2)
    
    # Rückgabe als Dictionary
    return {
        'mean': mean_values,
        'std': std_values
    }

OCV_mean={}
a=[]
s=[]
h=[]
for z in zellnamen:
    
    if "AM" in z: a.append(z)
    elif "dS24" in z: s.append(z)
    elif "HEL23" in z: h.append(z)
Zellarten={"AMNMC":a,"dS24NCA":s,"HEL23NMC":h}

for za in Zellarten.keys():
    OCV_mean[za]={}
    for mode in ["ch","dis"]:
        OCV_mean[za][mode]={}
        for grad in ["5°C","25°C","45°C"]:
            OCV_mean[za][mode][grad]={}
            for c in c_rate:
                OCV_mean[za][mode][grad][c]=mittelwert(OCV,Zellarten[za],mode,grad,c)
        
#%%
"""OCV mean dict speichern"""
with open('OCV_mean.pkl', 'wb') as file:
    pickle.dump(OCV_mean, file)
    
#%%
# plt.plot(OCV["AMNMC065"]["ch"]["5°C"]["C/5"][100:500,2],"--")
# plt.plot(inter[1][100:500,2],"--")
# plt.plot(test["mean"][100:500,2])

# colors=[ '#22a15c','#00a6b3','#cc0000', '#ff8000','#808080']
# # fig, axes = plt.subplots(3, 1, figsize=(8, 6))

# # Plot 1: OCV Daten
# temp=["25°C","45°C"]
# a=0
# savgol={"C/20":3001,"C/5":501,"C/2":201}
# for t in testnamen_OCV:
#     c=0

#     for i in c_rate:
#         ch = OCV_["ch"][t][i]
#         dis = OCV_["dis"][t][i]
#         dQ_ch = ch[:, 4] - min(ch[:, 4])
#         dQ_dis = dis[:, 4] - min(dis[:, 4])
    
#         plt.plot(dQ_ch, ch[:, 2], linestyle='--', color=colors[c])
#         plt.plot(dQ_dis, dis[:, 2], color=colors[c],label=temp[a]+" "+i)
#         c+=1
        
       
    
    # # Passe das Layout der Subplots an
    # plt.tight_layout()
    # plt.legend()
    # # Zeige den Plot
    # plt.show()
    # a+=1




