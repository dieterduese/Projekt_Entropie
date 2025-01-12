# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:54:50 2024

@author: maxim
"""

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


def consecutive(data, stepsize=3.5):
    return np.split(data, np.where(np.diff(data[:,0]) >= stepsize)[0]+1)

def consecutive_neu(data, stepsize=3.5):
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
    bounds_upper = [1, 50, 1, 200]  

    
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
                mod=np.array([UOCV,U0])
                if i == 0:
                    fit_ges = fit
                    IR=mod
                else:
                    fit_ges = np.column_stack([fit_ges, fit])
                    IR = np.column_stack([IR, mod])

        except RuntimeError as e:
            print(f"Curve fitting failed for pulse {i}: {e}")
            
    if laden==1:
        fit_ges[0,:]-=fit_ges[0,0]
        # SOC=fit_ges[0,:]/fit_ges[0,-1]
    elif laden==0:
        fit_ges[0,:]-=fit_ges[0,-1]
    else:
        print("Fehler bei Laderichtungsbestimmung")
    SOC_vol=np.array([fit_ges[0,:],IR[0,:]])
    
    
    return plot_ges, fit_ges,strom_split,IR, SOC_vol


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
#%%

"""OCV Daten erstellen"""

path=r"C:\Users\maxim\Nextcloud\Shared\Austausch_Max\Projekt_Entropie\OCV_Varianten"
zellnamen=os.listdir(path)
zellnamen=["dS24NCA01"]
c_rate=["C/5 iOCV","C/2 iOCV"]

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
    lines_iOCV_ch=[[45,46,47],[61,62,63]]
    lines_iOCV_dis=[[37,38,39],[53,54,55]] 
    
    
    OCV[z]={"ch":{},"dis":{}}
    
    
    for test in testnamen_OCV:
        if "_5deg" in test: t="5°C"
        elif "_25deg" in test: t="25°C"
        elif "_45deg" in test: t="45°C"
        
        l=0
        OCV[z]["ch"][t]={}
           
        for i in lines_iOCV_ch:
            tempch=filtplot(dict_test[test],i[0],i[2],plot=0,col=2)

            OCV[z]["ch"][t][c_rate[l]]=np.asarray(tempch,dtype=np.float64)
    
            l+=1
            
            
        l=0
        OCV[z]["dis"][t]={}
    
        for i in lines_iOCV_dis:
            tempdis=filtplot(dict_test[test],i[0],i[2],plot=0,col=2)

            OCV[z]["dis"][t][c_rate[l]]=np.asarray(tempdis,dtype=np.float64)
    
            l+=1

    fit_all=[]
    plot_all=[]
    hppc_data=[]
    IR_all=[]
    SOC_vol_all=[]
    for a in range(len(lines_iOCV_ch)):
        #Ladepulse(27,28)
        #Entladepulse(20,21)
        lines=[lines_iOCV_dis[a],lines_iOCV_ch[a]]
        data_t=dict_test["dS24NCA01_OCV_25deg.txt"]
        data_t = data_t.astype("float64")
        data_t[:,0]=data_t[:,0]*3600
        strome=[]
        for i in range(len(lines)):
            temp=filtplot(data_t,lines[i][1],lines[i][2],plot=0,col=3)
            temp=temp.astype(float)
            temp=consecutive_neu(temp) 
            # for i in range(len(temp)):
            #     temp[i]=process_array(temp[i])
            hppc_data.append(temp)
        plot_data,fit_data,IR_data,SOC_vol_data=[],[],[],[]
        for i in range(len(lines)):
            plot_temp,fit_temp,strom,IR,SOC_vol =hppc_fit_grenzen(data_t,lines[i],0)
            fit_temp=np.vstack([fit_temp])
            plot_data.append(plot_temp)
            fit_data.append(fit_temp)
            IR_data.append(IR)
            SOC_vol_data.append(SOC_vol)
        plot_all.append(plot_data)
        fit_all.append(fit_data)    
        IR_all.append(IR_data)
        SOC_vol_all.append(SOC_vol_data)
#%%
fig,axes=plt.subplots(2,2)
axes[0,0].plot(fit_all[0][0][0,:]/fit_all[0][0][0,0]*100,fit_all[0][0][2,:])
axes[1,0].plot(fit_all[0][0][0,:]/fit_all[0][0][0,0]*100,fit_all[0][0][3,:])
axes[0,1].plot(fit_all[0][0][0,:]/fit_all[0][0][0,0]*100,fit_all[0][0][4,:])
axes[1,1].plot(fit_all[0][0][0,:]/fit_all[0][0][0,0]*100,fit_all[0][0][5,:])
axes[0,0].plot(fit_all[0][1][0,:]/fit_all[0][1][0,-1]*100,fit_all[0][1][2,:])
axes[1,0].plot(fit_all[0][1][0,:]/fit_all[0][1][0,-1]*100,fit_all[0][1][3,:])
axes[0,1].plot(fit_all[0][1][0,:]/fit_all[0][1][0,-1]*100,fit_all[0][1][4,:])
axes[1,1].plot(fit_all[0][1][0,:]/fit_all[0][1][0,-1]*100,fit_all[0][1][5,:])
axes[0, 0].set_ylim(0, 0.05) 
axes[0, 1].set_ylim(0, 0.05)
plt.show()
# axes[1,0].set_ylim(0,10000)
#%%
d=10
plt.plot(np.linspace(hppc_data[0][d][0,0],hppc_data[0][d][-1,0],len(plot_all[0][0][d])),plot_all[0][0][d][:])
plt.plot(hppc_data[0][d][:,0],hppc_data[0][d][:,2]-hppc_data[0][d][0,2])
plt.show()
#%%
plt.plot(fit_all[0][0][0,:]/fit_all[0][0][0,0]*100,fit_all[0][0][1,:])
plt.plot(fit_all[0][1][0,:]/fit_all[0][1][0,-1]*100,fit_all[0][1][1,:])
plt.show()
#%%
"""anderes Modell"""

# moddata={"C/2 laden":{},"C/2 entladen":{},"C/5 laden":{},"C/5 entladen":{}}
# moddata["C/5 entladen"]["U"]=-IR_all[0][0]
# moddata["C/5 laden"]["U"]=IR_all[0][1]
# moddata["C/2 entladen"]["U"]=-IR_all[1][0]
# moddata["C/2 laden"]["U"]=IR_all[1][1]
# moddata["C/2 laden"]["I"]=1.56
# moddata["C/2 entladen"]["I"]=-1.56
# moddata["C/5 laden"]["I"]=0.624
# moddata["C/5 entladen"]["I"]=-0.624


# plt.scatter(moddata["C/5 entladen"]["U"][1,76],-0.624)
# plt.scatter(moddata["C/5 laden"]["U"][1,13],0.624)
# plt.scatter(0,0)
# plt.scatter(moddata["C/2 laden"]["U"][1,14],1.56)
# plt.scatter(moddata["C/2 entladen"]["U"][1,76],-1.56)
# #%%
# U_data=np.array([moddata["C/2 entladen"]["U"][1,84],moddata["C/5 entladen"]["U"][1,84],
#                  0,moddata["C/5 laden"]["U"][1,5],moddata["C/2 laden"]["U"][1,5]])
# #U_data=[-0.478901,-0.018507,0,0.0198429,0.048844]
# I_data=np.array([-1.56,-0.624,0,0.624,1.56])


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
