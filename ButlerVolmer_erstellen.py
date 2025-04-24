# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:03:01 2025

@author: maxim
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:55:59 2024

@author: maxim
"""
"""Funktionen"""


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
            UAh = dU/dAh
            if UAh!=np.inf:
                data.append([dataTime[i],UAh,dataAh[i]])
    data=np.array(data)
    return data

def ica(dataTime, dataU, dataAh):
    data = []
    for i in range(len(dataU)-1):
            dAh = dataAh[i+1] - dataAh[i]
            dU = dataU[i+1] - dataU[i]
            UAh = dAh/dU
            if UAh!=np.inf:
                data.append([dataTime[i],UAh,dataAh[i]])
    data=np.array(data)
    return data

def hppc(array,line):
    temp=array[array[:,1]==line]
    U1=temp[:,2]
    time=temp[:,0]
    # I=temp[:,3]
    
    return np.column_stack([time,U1])


def interp(ydata, xval):
    xq = np.arange(0,100,0.5)    

    x = np.array(xval, dtype=float)

    vq = interp1d(x, ydata)
    yint = vq(xq)
    return np.column_stack([xq,yint])


def consecutive(data, stepsize=3):
    return np.split(data, np.where(np.diff(data[:,0]) >= stepsize)[0]+1)

def consecutive_neu(data, stepsize=1.5):
    # Berechne Differenzen in der Zeitspalte (data[:, 0]) und in der Testspalte (data[:, 1])
    time_diff = np.diff(data[:, 0])
    test_diff = np.diff(data[:, 1])
    
    # Bedingung: Zeitdifferenz >= stepsize und Veränderung in der Testspalte
    split_indices = np.where((time_diff >= stepsize) & (test_diff != 0))[0] + 1
    
    # Array an den ermittelten Indizes aufteilen
    return np.split(data, split_indices)

def calcRi(data, datapoints):
    deltaU = data[datapoints-1,3] - data[datapoints,3]
    return abs(deltaU / data[datapoints-1,4])

def cost_func(initial_param,x,y,charge):
    J = sum(np.sqrt((y - model_func_par(x,charge,initial_param))**2))
    return J


    
def process_array(array):
    # Schritt 1: Umbruch zwischen Millisekunden- und Sekunden-Schritten finden
    diff = np.diff(array[:,0])
    break_point = np.where(diff > 0.1)[0][0] + 1  # +1, da diff eine Position weniger hat

    # Schritt 2: Werte vor dem Umbruch auf Sekundentakt zuschneiden
    pre_break_values = array[:break_point,:]
    pre_break_values = pre_break_values[::100]  # Nur jeden 100. Wert beibehalten, da 10ms -> 0.1s

    # Schritt 3: Werte nach dem Umbruch beibehalten
    post_break_values = array[break_point:,:]

    # Kombination der Arrays
    processed_array = np.concatenate((pre_break_values, post_break_values))

    return processed_array


def model_func_par(t, charge, param):
    U1, tau1, U2, tau2 = param
    if charge:
        return U1 * np.exp(-t/tau1) + U2 * np.exp(-t/tau2)
    else:
        return U1 * ( 1- np.exp(-t/tau1)) + U2 * (1 - np.exp(-t/tau2))
    
    
def hppc_fit(array,lines,strom):
    puls_temp=[]
    temp=filtplot(array,lines[1],lines[2],plot=0,col=2)
    U1_temp=temp[:,2]
    time_temp=temp[:,0]
    splitted_temp=np.column_stack([time_temp,U1_temp])
    splitted_temp=splitted_temp.astype(float)
    splitted=consecutive(splitted_temp)
    for i in range(len(splitted)):
        splitted[i]=process_array(splitted[i])
    
    strom_temp=filtplot(array,lines[0],lines[0],plot=0,col=3)
    strom_split=consecutive(strom_temp)
    strome=[]
    for i in range(len(strom_split)):
        if strom_split[i][-1,3]<0:
            laden=0
        else:
            laden=1
        strome.append(strom_split[i][-1,3]) 
        
    R0=[]
    fit_ges=np.array([])
    plot_ges=[]
    for i in range(len(splitted)):
        if i<len(strome):
            R0= abs((splitted[i][0,1]-strom_split[i][-1,2]) / strome[i])
        x=splitted[i][1:,0] - splitted[i][0,0] #delta Zeit als x-Werte
        x = x[:]
        
        if splitted[i][0,1] > splitted[i][-1,1]:
            y = splitted[i][1:,1] - splitted[i][-1,1] #delta U als y-Werte
        else:
            y = splitted[i][1:,1] - splitted[i][0,1]
        initial_param = np.array([0.1, 10, 0.2, 100])
       
        solution = fmin(cost_func,initial_param,args=(x,y,laden))  
        
        plotVal = model_func_par(x,laden, solution)
        plot_ges.append(plotVal)
        if i<len(strome):
            R1=solution[0]/abs(strome[i])
            C1=solution[1]/R1
            R2=solution[2]/abs(strome[i])
            C2=solution[3]/R2
            fit=np.array([R0,R1,C1,R2,C2])
    
            if i==0:
                fit_ges=fit
            else:
                fit_ges=np.column_stack([fit_ges,fit])
    return plot_ges,fit_ges

from scipy.optimize import curve_fit

def model_func(t, U1, tau1, U2, tau2, charge):
    if charge:
        return U1 * np.exp(-t / tau1) + U2 * np.exp(-t / tau2)
    else:
        return U1 * (1 - np.exp(-t / tau1)) + U2 * (1 - np.exp(-t / tau2))

def fit_with_curve_fit(x, y, charge):
    # Initiale Parameterwerte
    # initial_param = [0.05, 300, 0.05, 300] # [U1, tau1, U2, tau2]
    initial_param = [0.1, 10, 0.1, 100] 

    
    # Grenzen für die Parameter
    bounds_lower = [0, 1, 0, 1]  # Untere Grenzen: U1, tau1, U2, tau2
    # bounds_upper = [0.1, 600, 0.1, 600]  # Obere Grenzen: U1, tau1, U2, tau2
    bounds_upper = [1, 50, 1, 400]  

    
    # Anonyme Funktion für die Richtung (Laden/Entladen)
    def model_to_fit(t, U1, tau1, U2, tau2):
        return model_func(t, U1, tau1, U2, tau2, charge)
    
    # Curve Fit anwenden
    popt, pcov = curve_fit(
        model_to_fit, x, y, p0=initial_param, bounds=(bounds_lower, bounds_upper)
    )
    return popt, pcov

def hppc_fit_grenzen(array, lines, strom):
    puls_temp = []
    temp = filtplot(array, lines[1], lines[2], plot=0, col=2)
    # U1_temp = temp[:, 2]
    # time_temp = temp[:, 0]
    # cap_temp = temp[:,4]
    # splitted_temp = np.column_stack([time_temp, U1_temp,cap_temp])
    splitted_temp = temp.astype(float)
    splitted = consecutive_neu(splitted_temp)
    for i in range(len(splitted)):
        splitted[i] = process_array(splitted[i])

    strom_temp = filtplot(array, lines[0], lines[0], plot=0, col=3)
    strom_split = consecutive(strom_temp)
    strome = []
    for i in range(len(strom_split)):
        if strom_split[i][-1, 3] < 0:
            laden = 0
        else:
            laden = 1
        strome.append(strom_split[i][-1, 3])

    R0 = []
    U0 = []
    IR=np.array([])
    fit_ges = np.array([])
    plot_ges = []
    for i in range(len(splitted)):
        if i < len(strome):
            R0 = abs((splitted[i][0, 2] - strom_split[i][-1, 2]) / strome[i])
            U0 = abs((splitted[i][0, 2] - strom_split[i][-1, 2]))
            UOCV=splitted[i][-1,2]
        x = splitted[i][1:, 0] - splitted[i][0, 0]  # delta Zeit als x-Werte
        x = x[:]
        if splitted[i][0, 2] > splitted[i][-1,2]:
            y = splitted[i][1:, 2] - splitted[i][-1,2]  # delta U als y-Werte
        else:
            y = splitted[i][1:, 2] - splitted[i][0, 2]

        # Curve Fit anwenden
        try:
            solution, covariance = fit_with_curve_fit(x, y, laden)
            plotVal = model_func(x, solution[0], solution[1], solution[2], solution[3], laden)
            plot_ges.append(plotVal)

            if i < len(strome):
                cap=np.mean(splitted[i][:,4])
                R1 = solution[0] / abs(strome[i])
                Tau1 = solution[1] 
                R2 = solution[2] / abs(strome[i])
                Tau2 = solution[3] 
                fit = np.array([cap,R0, R1, Tau1, R2, Tau2])
                mod=np.array([cap,UOCV,U0])
                if i == 0:
                    fit_ges = fit
                    IR=mod
                else:
                    fit_ges = np.column_stack([fit_ges, fit])
                    IR = np.column_stack([IR, mod])

        except RuntimeError as e:
            print(f"Curve fitting failed for pulse {i}: {e}")
            
    if laden==1:
        IR[0,:]-=IR[0,0]
        fit_ges[0,:]-=fit_ges[0,0]
        # SOC=fit_ges[0,:]/fit_ges[0,-1]
    elif laden==0:
        IR[0,:]-=IR[0,-1]
        fit_ges[0,:]-=fit_ges[0,-1]
    else:
        print("Fehler bei Laderichtungsbestimmung")
    SOC_vol=np.array([fit_ges[0,:],IR[0,:]])
    
    
    return plot_ges, fit_ges,strom_split,IR


def cost_func2(initial_param,x,y):
    J = sum(np.sqrt((y - model2(x,y,initial_param))**2))
    return J

def model2(U0,I, param):
    I0, alpha, R = param
    z = 1  # Anzahl der übertragenen Elektronen (z.B. für Li+)
    F = 96485.3329  # Faraday-Konstante (C/mol)
    R_const = 8.3145  # Universelle Gaskonstante (J/mol/K)
    T = 298.15  # Absolute Temperatur (in Kelvin)

    # Aktivierungsüberpotentialterm
    hct = U0 - I*R

    # Butler-Volmer-Gleichung
    return I0 * (
        np.exp((alpha * z * F / (R_const * T)) * hct) -
        np.exp(-(1 - alpha) * z * F / (R_const * T) * hct)
    )
def butler_volmer_fit(testdictV):
    butler_fit_t=np.array([])
    butler_plot_t={}
    for a in range(len(testdictV["C/2 ch"]["U"][0,:])):
        Udata=[0]
        Idata=[0]
        for i in testdictV.keys():
            if "5.1A" not in i and "1.5C" not in i:
                Udata.append(testdictV[i]["U"][1,a])
                Idata.append(testdictV[i]["I"])
                print(i)
        Udata=np.sort(np.array(Udata))
        Idata=np.sort(np.array(Idata))
        p0 = [1.5, 0.5, 0.01]  # Startwerte: I0, alpha, IR
        # bounds = (
        #     [0, 0.1, 0.01],  # Untergrenzen: I0, alpha, IR
        #     [1.0, 2, 10],     # Obergrenzen: I0, alpha, IR
        # )

        # popt, pcov = curve_fit(
        #     model2, I_data, U_data,
        #     p0=p0, bounds=bounds
        # )
        solution = fmin(cost_func2,p0,args=(Udata,Idata))  
        # Ausgabe der Fit-Ergebnisse
        I0_fit, alpha_fit, R_fit = solution
        print(f"Fit-Ergebnisse:\nI0: {I0_fit:.4e}\nalpha: {alpha_fit:.4f}\nR: {R_fit:.4f}")

        # Fit-Kurve berechnen
        Ufit = np.linspace(min(Udata), max(Udata), 100)
        Ifit=np.linspace(min(Idata),max(Idata),100)
        Ifitted = model2(Ufit,Ifit, solution)
        
        fit_t=np.array([testdictV["C/2 ch"]["U"][0,a],I0_fit, alpha_fit, R_fit])
        plot_t=np.array([Ufit,Ifitted]).T
        if a == 0:
            butler_fit_t = fit_t
        else:
            butler_fit_t = np.column_stack([butler_fit_t, fit_t])
        vol=testdictV["C/2 ch"]["U"][0,a]
        butler_plot_t[f"{vol:.4f}"]=plot_t 
        
        
    return butler_fit_t,butler_plot_t
#%%
"""Daten laden"""
# initpath=r"C:\Users\maxim\Nextcloud\Shared\Austausch_Max\Projekt_Entropie\OCV_daten"

initpath=r"C:\Users\Dominik\tubCloud\Studis\Austausch_Max\Projekt_Entropie\OCV_daten"
datei="dS24NCA10_ButlerVolmerGITT.txt"
datei2="dS24NCA10_ButlerVolmerGITT_lowI.txt"
data_raw=load(initpath,datei,skip=0,separation=" ")
data_temp=np.delete(data_raw,[1,2,3,5,10,12,13],1)
data=data_temp
data_raw2=load(initpath,datei2,skip=0,separation=" ")
data_temp2=np.delete(data_raw2,[1,2,3,5,10,12,13],1)
data2=data_temp2
#%%
test=filtplot(data,0,109,plot=1,col=4)
#%%
lines_iOCV_dis=[[19,20,21],[35,36,37],[51,52,53],[67,68,69],[83,84,85],[99,100,101]]
lines_iOCV_ch=[[27,28,29],[43,44,45],[59,60,61],[75,76,77],[91,92,93],[107,108,109]]
OCV={"ch":{},"dis":{}}
c_rate=["C/5","C/2","3C/4","1C","1.5C","5.1A"]
c_rate2=["C/20","C/10"]


l=0
   
for i in lines_iOCV_ch:
    tempch=filtplot(data,i[0],i[2],plot=0,col=2)

    OCV["ch"][c_rate[l]]=np.asarray(tempch,dtype=np.float64)

    l+=1
    
    
l=0
for i in lines_iOCV_ch[:2]:
    tempch=filtplot(data2,i[0],i[2],plot=0,col=2)

    OCV["ch"][c_rate2[l]]=np.asarray(tempch,dtype=np.float64)

    l+=1
    
l=0 
for i in lines_iOCV_dis:
    tempdis=filtplot(data,i[0],i[2],plot=0,col=2)

    OCV["dis"][c_rate[l]]=np.asarray(tempdis,dtype=np.float64)

    l+=1
    
l=0
for i in lines_iOCV_dis[:2]:
    tempdis=filtplot(data2,i[0],i[2],plot=0,col=2)

    OCV["dis"][c_rate2[l]]=np.asarray(tempdis,dtype=np.float64)

    l+=1
       
#%%
fit_all={}
plot_all={}
hppc_data=[]
IR_all={}
l=0
for a in range(len(lines_iOCV_ch)):
    #Ladepulse(27,28)
    #Entladepulse(20,21)
    lines=[lines_iOCV_dis[a],lines_iOCV_ch[a]]
    data_t=data.copy()
    data_t[:,0]=data_t[:,0]*3600
    strome=[]
    for i in range(len(lines)):
        temp=filtplot(data_t,lines[i][1],lines[i][2],plot=0,col=3)
        temp=temp.astype(float)
        temp=consecutive_neu(temp) 
        # for i in range(len(temp)):
        #     temp[i]=process_array(temp[i])
        hppc_data.append(temp)
    plot_data,fit_data,IR_data=[],[],[]
    for i in range(len(lines)):
        plot_temp,fit_temp,strom,IR =hppc_fit_grenzen(data_t,lines[i],0)
        fit_temp=np.vstack([fit_temp])
        plot_data.append(plot_temp)
        fit_data.append(fit_temp)
        IR_data.append(IR)
    plot_all[c_rate[l]]=plot_data
    fit_all[c_rate[l]]=fit_data    
    IR_all[c_rate[l]]=IR_data
    l+=1
l=0    
for a in range(len(lines_iOCV_ch[:2])):
    #Ladepulse(27,28)
    #Entladepulse(20,21)
    lines=[lines_iOCV_dis[a],lines_iOCV_ch[a]]
    data_t=data2.copy()
    data_t[:,0]=data_t[:,0]*3600
    strome=[]
    for i in range(len(lines)):
        temp=filtplot(data_t,lines[i][1],lines[i][2],plot=0,col=3)
        temp=temp.astype(float)
        temp=consecutive_neu(temp) 
        # for i in range(len(temp)):
        #     temp[i]=process_array(temp[i])
        hppc_data.append(temp)
    plot_data,fit_data,IR_data=[],[],[]
    for i in range(len(lines)):
        plot_temp,fit_temp,strom,IR =hppc_fit_grenzen(data_t,lines[i],0)
        fit_temp=np.vstack([fit_temp])
        plot_data.append(plot_temp)
        fit_data.append(fit_temp)
        IR_data.append(IR)
    plot_all[c_rate2[l]]=plot_data
    fit_all[c_rate2[l]]=fit_data    
    IR_all[c_rate2[l]]=IR_data
    l+=1

#%%
c="C/5"
fig,axes=plt.subplots(2,2,figsize=(10, 7))
axes[0,0].plot(fit_all[c][0][0,:]/fit_all[c][0][0,0]*100,fit_all[c][0][2,:],label="dis")
axes[1,0].plot(fit_all[c][0][0,:]/fit_all[c][0][0,0]*100,fit_all[c][0][3,:])
axes[0,1].plot(fit_all[c][0][0,:]/fit_all[c][0][0,0]*100,fit_all[c][0][4,:])
axes[1,1].plot(fit_all[c][0][0,:]/fit_all[c][0][0,0]*100,fit_all[c][0][5,:])
axes[0,0].plot(fit_all[c][1][0,:]/fit_all[c][1][0,-1]*100,fit_all[c][1][2,:],label="ch")
axes[1,0].plot(fit_all[c][1][0,:]/fit_all[c][1][0,-1]*100,fit_all[c][1][3,:])
axes[0,1].plot(fit_all[c][1][0,:]/fit_all[c][1][0,-1]*100,fit_all[c][1][4,:])
axes[1,1].plot(fit_all[c][1][0,:]/fit_all[c][1][0,-1]*100,fit_all[c][1][5,:])
axes[0, 0].set_ylim(0, 0.05) 
axes[0, 1].set_ylim(0, 0.05)
axes[0,0].set_title("R1")
axes[0,1].set_title("R2")
axes[1,0].set_title("C1")
axes[1,1].set_title("C2")
axes[0,0].legend()
plt.show()
#%%
c="C/2"
ylims={"C/20":0.05,"C/10":0.05,"C/5":0.05, "C/2":0.04,
       "5.1A":0.012,"1.5C":0.012,"1C":0.018,"3C/4":0.025}

for c in fit_all.keys():
    fig,axes=plt.subplots(2,2,figsize=(10, 7))
    axes[0,0].plot(IR_all[c][0][0,:],fit_all[c][0][2,:],label="dis")
    axes[1,0].plot(IR_all[c][0][0,:],fit_all[c][0][3,:])
    axes[0,1].plot(IR_all[c][0][0,:],fit_all[c][0][4,:])
    axes[1,1].plot(IR_all[c][0][0,:],fit_all[c][0][5,:])
    axes[0,0].plot(IR_all[c][1][0,:],fit_all[c][1][2,:],label="ch")
    axes[1,0].plot(IR_all[c][1][0,:],fit_all[c][1][3,:])
    axes[0,1].plot(IR_all[c][1][0,:],fit_all[c][1][4,:])
    axes[1,1].plot(IR_all[c][1][0,:],fit_all[c][1][5,:])
    axes[0,0].legend()
    axes[0,0].set_title("R1")
    axes[0,1].set_title("R2")
    axes[1,0].set_title("C1")
    axes[1,1].set_title("C2")
    axes[0, 0].set_ylim(0, ylims[c]) 
    axes[0, 1].set_ylim(0, ylims[c])
    fig.suptitle(c, fontsize=14, fontweight='bold', y=0.95)
    plt.show()
#%%
"""Alle Fits ein Plot"""
col = ['#22a15c', '#00a6b3', '#cc0000', '#ff8000', '#808080', 'yellow', '#6a0dad', '#8B4513']
i=0
fig,axes=plt.subplots(2,2,figsize=(10, 7))
for c in fit_all.keys():
    axes[0,0].plot(IR_all[c][1][0,:],fit_all[c][0][2,:],color=col[i],label=c)
    axes[1,0].plot(IR_all[c][1][0,:],fit_all[c][0][3,:],color=col[i])
    axes[0,1].plot(IR_all[c][1][0,:],fit_all[c][0][4,:],color=col[i])
    axes[1,1].plot(IR_all[c][1][0,:],fit_all[c][0][5,:],color=col[i])

    i+=1
axes[0,0].legend()
axes[0,0].set_title("R1")
axes[0,1].set_title("R2")
axes[1,0].set_title("C1")
axes[1,1].set_title("C2")
axes[0, 0].set_ylim(0, 0.03) 
axes[0, 1].set_ylim(0, 0.03)
fig.suptitle("Entladerichtung", fontsize=14, fontweight='bold', y=0.95)
plt.show()


i=0
fig,axes=plt.subplots(2,2,figsize=(10, 7))
for c in fit_all.keys():

    axes[0,0].plot(IR_all[c][1][0,:],fit_all[c][1][2,:],color=col[i],label=c)
    axes[1,0].plot(IR_all[c][1][0,:],fit_all[c][1][3,:],color=col[i])
    axes[0,1].plot(IR_all[c][1][0,:],fit_all[c][1][4,:],color=col[i])
    axes[1,1].plot(IR_all[c][1][0,:],fit_all[c][1][5,:],color=col[i])
    i+=1
axes[0,0].legend()
axes[0,0].set_title("R1")
axes[0,1].set_title("R2")
axes[1,0].set_title("C1")
axes[1,1].set_title("C2")
axes[0, 0].set_ylim(0, 0.03) 
axes[0, 1].set_ylim(0, 0.03)
fig.suptitle("Laderichtung", fontsize=14, fontweight='bold', y=0.95)
plt.show()



#%%
d=90
c1=14
c2="C/10"
plt.plot(np.linspace(hppc_data[c1][d][0,0],hppc_data[c1][d][-1,0],len(plot_all[c2][0][d])),plot_all[c2][0][d][:])
plt.plot(hppc_data[c1][d][:,0],hppc_data[c1][d][:,2]-hppc_data[c1][d][0,2])
plt.show()
#%%
"""Butler-Volmer"""
moddata={}
for direction in OCV.keys():
    
    for c in OCV[direction].keys():
        moddata[f"{c}"+" "+f"{direction}"]={}
        if direction=="ch":
            moddata[f"{c}"+" "+f"{direction}"]["U"]=IR_all[c][1]
        elif direction=="dis":
            moddata[f"{c}"+" "+f"{direction}"]["U"]=np.array([IR_all[c][0][0,:],IR_all[c][0][1,:],-IR_all[c][0][2,:]])
        else:
            print("falsche Laderichtung")
        moddata[f"{c}"+" "+f"{direction}"]["I"]=OCV[direction][c][5,3]
        # moddata[f"{c}"+" "+f"{direction}"]["I"]=np.mean(OCV[direction][c][1:10,3])

        

#%%
import copy

testdict = copy.deepcopy(moddata)
testdictU=copy.deepcopy(testdict)
testdictQ=copy.deepcopy(testdict)

minvol=[]
maxvol=[]  
mincap=[]
maxcap=[]  
for i in testdict.keys():
    mincap.append(min(testdict[i]["U"][0,:]))
    maxcap.append(max(testdict[i]["U"][0,:]))
    minvol.append(min(testdict[i]["U"][1,:]))
    maxvol.append(max(testdict[i]["U"][1,:]))
commonvol=np.linspace(max(minvol),min(maxvol),100)
commoncap=np.linspace(max(mincap),min(maxcap),100)
for i in testdict.keys():
    
    f1U=interp1d(testdict[i]["U"][1,:],testdict[i]["U"][2,:],kind='linear')
    interpU=f1U(commonvol)
    testdictU[i]["U"]=np.array([commonvol,interpU])
    f1Q=interp1d(testdict[i]["U"][0,:],testdict[i]["U"][2,:],kind='linear')
    interpQ=f1Q(commoncap)
    testdictQ[i]["U"]=np.array([commoncap,interpQ])
#%%
# Butler Volmer testen
# e=10
# U_data=np.array([testdict2["1C dis"]["U"][1,e],
#                  testdict2["3C/4 dis"]["U"][1,e],testdict2["C/2 dis"]["U"][1,e],testdict2["C/5 dis"]["U"][1,e],
#                  0,
#                  testdict2["C/5 ch"]["U"][1,e],testdict2["C/2 ch"]["U"][1,e],testdict2["3C/4 ch"]["U"][1,e],
#                  testdict2["1C ch"]["U"][1,e]])
# #U_data=[-0.478901,-0.018507,0,0.0198429,0.048844]
# I_data=np.array([testdict2["1C dis"]["I"],
#                  testdict2["3C/4 dis"]["I"],testdict2["C/2 dis"]["I"],testdict2["C/5 dis"]["I"],
#                  0,
#                  testdict2["C/5 ch"]["I"],testdict2["C/2 ch"]["I"],testdict2["3C/4 ch"]["I"],
#                 testdict2["1C ch"]["I"]])


# p0 = [1.5, 0.5, 0.01]  # Startwerte: I0, alpha, IR
# # bounds = (
# #     [0, 0.1, 0.01],  # Untergrenzen: I0, alpha, IR
# #     [1.0, 2, 10],     # Obergrenzen: I0, alpha, IR
# # )

# # popt, pcov = curve_fit(
# #     model2, I_data, U_data,
# #     p0=p0, bounds=bounds
# # )
# solution = fmin(cost_func2,p0,args=(U_data,I_data))  
# # Ausgabe der Fit-Ergebnisse
# I0_fit, alpha_fit, R_fit = solution
# print(f"Fit-Ergebnisse:\nI0: {I0_fit:.4e}\nalpha: {alpha_fit:.4f}\nR: {R_fit:.4f}")

# # Fit-Kurve berechnen
# U_fit = np.linspace(min(U_data), max(U_data), 100)
# I_fit=np.linspace(min(I_data),max(I_data),100)
# I_fit = model2(U_fit,I_fit, solution)

# # Plot der Daten und des Fits
# plt.scatter( U_data,I_data, label="Daten", color="blue")
# plt.plot( U_fit,I_fit ,label="Fit", color="red")

# plt.legend()
# plt.grid()
# plt.title("Curve-Fit mit modifizierter Butler-Volmer-Gleichung")
# plt.show()
#%%
"""Butler Volmer Fit über Spannung"""
butler_fit={"Kapazität":{},"Spannung":{}}
butler_plot={"Kapazität":{},"Spannung":{}}
butler_fit["Spannung"],butler_plot["Spannung"]=butler_volmer_fit(testdictU)
butler_fit["Kapazität"],butler_plot["Kapazität"]=butler_volmer_fit(testdictQ)

# for a in range(len(testdictU["C/2 ch"]["U"][0,:])):
#     Udata=[0]
#     Idata=[0]
#     for i in testdictU.keys():
#         if "5.1A" not in i and "1.5C" not in i:
#             Udata.append(testdictU[i]["U"][1,a])
#             Idata.append(testdictU[i]["I"])
#             print(i)
#     Udata=np.sort(np.array(Udata))
#     Idata=np.sort(np.array(Idata))
#     p0 = [1.5, 0.5, 0.01]  # Startwerte: I0, alpha, IR
#     # bounds = (
#     #     [0, 0.1, 0.01],  # Untergrenzen: I0, alpha, IR
#     #     [1.0, 2, 10],     # Obergrenzen: I0, alpha, IR
#     # )

#     # popt, pcov = curve_fit(
#     #     model2, I_data, U_data,
#     #     p0=p0, bounds=bounds
#     # )
#     solution = fmin(cost_func2,p0,args=(Udata,Idata))  
#     # Ausgabe der Fit-Ergebnisse
#     I0_fit, alpha_fit, R_fit = solution
#     print(f"Fit-Ergebnisse:\nI0: {I0_fit:.4e}\nalpha: {alpha_fit:.4f}\nR: {R_fit:.4f}")

#     # Fit-Kurve berechnen
#     Ufit = np.linspace(min(Udata), max(Udata), 100)
#     Ifit=np.linspace(min(Idata),max(Idata),100)
#     Ifitted = model2(Ufit,Ifit, solution)
    
#     fit_t=np.array([testdictU["C/2 ch"]["U"][0,a],I0_fit, alpha_fit, R_fit])
#     plot_t=np.array([Ufit,Ifitted]).T
#     if a == 0:
#         butler_fit_t = fit_t
#     else:
#         butler_fit_t = np.column_stack([butler_fit_t, fit_t])
#     vol=testdictU["C/2 ch"]["U"][0,a]
#     butler_plot["Spannung"][f"{vol:.4f}"+" V"]=plot_t 
# butler_fit["Spannung"]=butler_fit_t
#%%
e = 10
U = np.array([testdictU["1C dis"]["U"][1, e],
              testdictU["3C/4 dis"]["U"][1, e], testdictU["C/2 dis"]["U"][1, e], testdictU["C/5 dis"]["U"][1, e],
              testdictU["C/10 dis"]["U"][1, e], testdictU["C/20 dis"]["U"][1, e],
              0,
              testdictU["C/20 ch"]["U"][1, e], testdictU["C/10 ch"]["U"][1, e],
              testdictU["C/5 ch"]["U"][1, e], testdictU["C/2 ch"]["U"][1, e], testdictU["3C/4 ch"]["U"][1, e],
              testdictU["1C ch"]["U"][1, e]])

I = np.array([testdictU["1C dis"]["I"],
              testdictU["3C/4 dis"]["I"], testdictU["C/2 dis"]["I"], testdictU["C/5 dis"]["I"],
              testdictU["C/10 dis"]["I"], testdictU["C/20 dis"]["I"],
              0,
              testdictU["C/20 ch"]["I"], testdictU["C/10 ch"]["I"],
              testdictU["C/5 ch"]["I"], testdictU["C/2 ch"]["I"], testdictU["3C/4 ch"]["I"],
              testdictU["1C ch"]["I"]])

plt.scatter(U, I, label="Daten", color="blue")
plt.plot(butler_plot["Spannung"]["3.0815"][:, 0], butler_plot["Spannung"]["3.0815"][:, 1], label="Fit 3.0815V", color="red")

plt.xlabel("$U_{R}(V)$")
plt.ylabel("I(A)")
plt.legend()
plt.grid()
plt.ylabel("I (A)")
plt.xlabel("$\eta$ (V)")
plt.title("Curve-Fit mit modifizierter Butler-Volmer-Gleichung")
plt.show()

#%%
"""Parameter plotten"""
colors = ['#22a15c', '#00a6b3', '#cc0000', '#ff8000', '#808080',"yellow"]
for x in butler_fit.keys():
# alpha
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)  # 3 Zeilen, 1 Spalte
    
    # Erster Plot
    axes[0].plot(butler_fit[x][0, :], butler_fit[x][1, :],color=colors[0])
    axes[0].grid()
    axes[0].set_ylabel("$I_{0}$")
    
    # Zweiter Plot
    axes[1].plot(butler_fit[x][0, :], butler_fit[x][2, :],color=colors[1])
    axes[1].grid()
    axes[1].set_ylabel("α")
    
    # Dritter Plot
    axes[2].plot(butler_fit[x][0, :], butler_fit[x][3, :],color=colors[2])
    axes[2].grid()
    if x=="Spannung":
        axes[2].set_xlabel("U(V)")
    elif x=="Kapazität":
        axes[2].set_xlabel("Q(Ah)")
    else:
        print("Fehler: unbekannte x-Achse")

        
    axes[2].set_ylabel("R")
    
    plt.tight_layout()  # Optimiert die Abstände zwischen den Plots
    plt.show()


#%%
a = "C/10 dis"
plt.plot(testdict[a]["U"][0, :], testdict[a]["U"][1, :])
plt.plot(testdictU[a]["U"][0, :], testdictU[a]["U"][1, :])
#%%
e = 10
U = np.array([testdictQ["1C dis"]["U"][1, e],
              testdictQ["3C/4 dis"]["U"][1, e], testdictQ["C/2 dis"]["U"][1, e], testdictQ["C/5 dis"]["U"][1, e],
              testdictQ["C/10 dis"]["U"][1, e], testdictQ["C/20 dis"]["U"][1, e],
              0,
              testdictQ["C/20 ch"]["U"][1, e], testdictQ["C/10 ch"]["U"][1, e],
              testdictQ["C/5 ch"]["U"][1, e], testdictQ["C/2 ch"]["U"][1, e], testdictQ["3C/4 ch"]["U"][1, e],
              testdictQ["1C ch"]["U"][1, e]])

I = np.array([testdictQ["1C dis"]["I"],
              testdictQ["3C/4 dis"]["I"], testdictQ["C/2 dis"]["I"], testdictQ["C/5 dis"]["I"],
              testdictQ["C/10 dis"]["I"], testdictQ["C/20 dis"]["I"],
              0,
              testdictQ["C/20 ch"]["I"], testdictQ["C/10 ch"]["I"],
              testdictQ["C/5 ch"]["I"], testdictQ["C/2 ch"]["I"], testdictQ["3C/4 ch"]["I"],
              testdictQ["1C ch"]["I"]])

plt.scatter(U, I, label="Daten", color="blue")
plt.plot(butler_plot["Kapazität"]["0.2430"][:, 0], butler_plot["Kapazität"]["0.2430"][:, 1], label="Fit 0.2430Ah", color="red")

plt.xlabel("$U_{R}(V)$")
plt.ylabel("I(A)")
plt.legend()
plt.grid()
plt.title("Curve-Fit mit modifizierter Butler-Volmer-Gleichung")
plt.show()

#%%
# U_data=np.array([moddata["5.1A dis"]["U"][1,82],moddata["1.5C dis"]["U"][1,82],moddata["1C dis"]["U"][1,82],
#                   moddata["3C/4 dis"]["U"][1,82],moddata["C/2 dis"]["U"][1,82],moddata["C/5 dis"]["U"][1,82],
#                   0,
#                   moddata["C/5 ch"]["U"][1,5],moddata["C/2 ch"]["U"][1,5],moddata["3C/4 ch"]["U"][1,5],
#                   moddata["1C ch"]["U"][1,5],moddata["1.5C ch"]["U"][1,5],moddata["5.1A ch"]["U"][1,5]])
# #U_data=[-0.478901,-0.018507,0,0.0198429,0.048844]
# I_data=np.array([moddata["5.1A dis"]["I"],moddata["1.5C dis"]["I"],moddata["1C dis"]["I"],
#                   moddata["3C/4 dis"]["I"],moddata["C/2 dis"]["I"],moddata["C/5 dis"]["I"],
#                   0,
#                   moddata["C/5 ch"]["I"],moddata["C/2 ch"]["I"],moddata["3C/4 ch"]["I"],
#                   moddata["1C ch"]["I"],moddata["1.5C ch"]["I"],moddata["5.1A ch"]["I"]])


# p0 = [0.4, 0.3, 0.01]  # Startwerte: I0, alpha, IR
# # bounds = (
# #     [0, 0.1, 0.01],  # Untergrenzen: I0, alpha, IR
# #     [1.0, 2, 10],     # Obergrenzen: I0, alpha, IR
# # )

# # popt, pcov = curve_fit(
# #     model2, I_data, U_data,
# #     p0=p0, bounds=bounds
# # )
# solution = fmin(cost_func2,p0,args=(U_data,I_data))  
# # Ausgabe der Fit-Ergebnisse
# I0_fit, alpha_fit, R_fit = solution
# print(f"Fit-Ergebnisse:\nI0: {I0_fit:.4e}\nalpha: {alpha_fit:.4f}\nR: {R_fit:.4f}")

# # Fit-Kurve berechnen
# U_fit = np.linspace(min(U_data), max(U_data), 100)
# I_fit=np.linspace(min(I_data),max(I_data),100)
# I_fit = model2(U_fit,I_fit, solution)

# # Plot der Daten und des Fits
# plt.scatter( U_data,I_data, label="Daten", color="blue")
# plt.plot( U_fit,I_fit ,label="Fit", color="red")

# plt.legend()
# plt.grid()
# plt.title("Curve-Fit mit modifizierter Butler-Volmer-Gleichung")
# plt.show()