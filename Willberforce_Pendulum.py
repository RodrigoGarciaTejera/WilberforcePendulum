#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:45:08 2022

@author: Gallot thomas
"""
#import necessary modules
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import integrate
import matplotlib.pyplot as plt
import csv
import tkinter as tk
import scipy.signal as signal
import scipy.optimize

root = tk.Tk()
root.withdraw()


def readcsv(filename):
    Data = pd.read_csv(filename,delimiter=';', skiprows=1) #Please add four spaces here before this line
    Data=np.array(Data) #Please add four spaces here before this line
    Data = np.vstack(Data[1:,0:-1]).astype(np.float)
    Time=Data[:,-1]/1000
    Data=Data[:,0:-1]
    with open(filename) as f:
        reader = csv.reader(f)
        headings = next(reader)
        headings = next(reader)
    headings=str(headings).split(";")
    return [Time,Data,headings]
    [Time,Data,headings] = readcsv(yourFilename)
    
def ReadData(Nfile):
    filename=filenameAll[Nfile]
    [Time,Data,headings] =readcsv(filename+'.csv')

    Time=Time[int(Ini[Nfile]-Nh):int(Fin[Nfile])]-Time[int(Ini[Nfile])]
    Data= Data[int(Ini[Nfile]-Nh):int(Fin[Nfile]),:]
    Data=Data-Data.mean(axis = 0)
    dt=np.mean(np.diff(Time))
    fs=1/dt
    Nt=Time.shape[0]
    H= np.blackman(Nh*2);
    W=np.concatenate((H[0:Nh],np.ones(Nt-Nh*2),H[Nh:]),axis=0)
    Data=MtimesV(Data,W)
    Data=Data-Data.mean(axis = 0)
    #Data=signal.detrend(Data[Ini:Fin,:], axis=0)
    W1=[0.1,1]
    sos= signal.butter(5, W1,fs=fs,btype='bandpass',output='sos')
    Data=signal.sosfiltfilt(sos, Data,axis=0)  
    z_acc=-Data[:,2];# the celphone is looking done so z-axis absolute is oposite to z-axis cellphone
    z_vel=signal.detrend(integrate.cumtrapz(z_acc,Time,initial=0))
    z=signal.detrend(integrate.cumtrapz(signal.detrend(integrate.cumtrapz(z_acc,Time,initial=0)),Time,initial=0))
    theta_vel=-Data[:,8]; # the celphone is looking done so z-axis absolute is oposite to z-axis cellphone
    theta=signal.detrend( integrate.cumtrapz(theta_vel,Time,initial=0))
    theta_acc=np.gradient(theta_vel,Time);
    theta_acc=signal.detrend(theta_acc);
    Npad=10
    Nf=Npad*z_acc[Nh:].shape[0]
    z_accTF=np.fft.fft(z_acc[Nh:],axis=0,n=Nf)
    theta_accTF=np.fft.fft(theta_acc[Nh:],axis=0,n=Nf)
    theta_velTF=np.fft.fft(theta_vel[Nh:],axis=0,n=Nf)
    Freq=np.fft.fftfreq(Nf,dt)   
    return z_acc, theta_acc, z_vel, theta_vel, z, theta, theta_accTF, z_accTF, Freq, Time, Nf
    
    
    
def plotyy(t,data1,data2,col1='red',col2='k',xlabel='time (s)',ylabel1='data1',ylabel2='data2',t2=np.array([0])):
    color = 'tab:' + col1
    fig,ax1=plt.subplots(1,1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim([np.amin(t),np.amax(t)])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = col2
    ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label with ax1
    if t2.shape[0]<2:
        ax2.plot(t, data2, color=color)
    else :
        ax2.plot(t2, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return [ax1,ax2,fig]


def plotyyDbl(t,data1,data2,Dt,Ddata1,Ddata2,col1='red',col2='k',xlabel1='Time (s)',xlabel2='Time (s)',ylabel1='data1',ylabel2='data2',ylabel3='ylabel1',ylabel4='ylabel2',t2=np.array([0]),Dt2=np.array([0])):
    color = 'tab:' + col1
    fig,ax1=plt.subplots(2,1)
    ax1[0].set_xlabel(xlabel1)
    ax1[0].set_ylabel(ylabel1, color=color)
    ax1[0].plot(t, data1, color=color)
    ax1[0].tick_params(axis='y', labelcolor=color)
    ax1[0].set_xlim([np.amin(t),np.amax(t)])
       
    ax1[1].set_xlabel(xlabel2)
    ax1[1].set_ylabel(ylabel3, color=color)
    ax1[1].plot(Dt, Ddata1, color=color)
    ax1[1].tick_params(axis='y', labelcolor=color)
    ax1[1].set_xlim([np.amin(t),np.amax(t)])
    
    ax2 = ax1[0].twinx()  # instantiate a second axes that shares the same x-axis
    color = col2
    ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label with ax1
    if t2.shape[0]<2:
        ax2.plot(t, data2, color=color)
    else :
        ax2.plot(t2, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)  
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    ax3 = ax1[1].twinx()  # instantiate a second axes that shares the same x-axis
    
    ax3.set_ylabel(ylabel4, color=color)  # we already handled the x-label with ax1
    if Dt2.shape[0]<2:
        ax3.plot(Dt, Ddata2, color=color)
    else :
        ax3.plot(Dt2, Ddata2, color=color)
 
    ax3.tick_params(axis='y', labelcolor=color) 
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
   
    return [ax1,ax2,ax3,fig]   



def plotDbl(t,data1,data2,Dt,Ddata1,Ddata2,col1='k',col2='r',xlabel1='Time (s)',xlabel2='Time (s)',ylabel1='data1',ylabel2='data2',t2=np.array([0]),Dt2=np.array([0]),linestyle1='-',linestyle2='-'):
    color = col1
    fig,ax1=plt.subplots(2,1)
    ax1[0].set_xlabel(xlabel1)
    ax1[0].set_ylabel(ylabel1)
    ax1[0].plot(t, data1, color=color,linestyle=linestyle1)
    #ax1[0].tick_params(axis='y', labelcolor=color)
    ax1[0].set_xlim([np.amin(t),np.amax(t)])
    
    color = col2
    ax1[1].set_xlabel(xlabel2)
    ax1[1].set_ylabel(ylabel2)
    ax1[1].plot(Dt, Ddata1, color=color,linestyle=linestyle1)
    #ax1[1].tick_params(axis='y', labelcolor=color)
    color = col1
    ax1[1].set_xlim([np.amin(t),np.amax(t)])
    if t2.shape[0]<2:
        ax1[0].plot(t, data2, color=color,linestyle=linestyle2)
    else :
        ax1[0].plot(t2, data2, color=color,linestyle=linestyle2)
    fig.tight_layout() 
    color = col2
    if Dt2.shape[0]<2:
        ax1[1].plot(Dt, Ddata2, color=color,linestyle=linestyle2)
    else :
        ax1[1].plot(Dt2, Ddata2, color=color,linestyle=linestyle2)
    plt.show()
    fig.tight_layout() 
    return [ax1,fig]   

    
def MtimesV(M,V):
    if M.shape[0]==len(V):
        VV=np.tile(V,(M.shape[1],1)).T
        MtimesV=np.multiply(M,VV)
    if M.shape[1]==len(V):
        VV=np.tile(V,(M.shape[0],1))
        MtimesV=np.multiply(M,VV)
    return MtimesV


def derivAtt(y, t, omega, epsilon, m, I,alpha):
    """Return the first derivatives of y = z, zdot, theta, thetadot."""
    z, zdot, theta, thetadot = y
    dzdt = zdot
    dzdotdt =-alpha*zdot -omega**2 * z - epsilon / 2 / m * theta
    dthetadt = thetadot
    dthetadotdt =-alpha*thetadot -omega**2 * theta - epsilon / 2 / I * z
    return dzdt, dzdotdt, dthetadt, dthetadotdt



filenameAll  =['bat1' ,'bat2' ,'modo1' ,'modo2']



#heading point number
Nh=100

#initial and final point for each data
Ini=np.array([ 476., 1093.,  240.,  164.])
Fin=np.array([1544., 2576.,  818.,  856.])

##################################################    
# Pulsation definitions (Estimated from ffts)
################################################
f1=0.4071685
f2=0.53919023
omega1=2*np.pi*f1
omega2=2*np.pi*f2
omega=((omega2**2+omega1**2)/2)**0.5
omegaB=(omega2**2-omega1**2)/2/omega


##################################################    
# Geometrical Data (in meter)
################################################
Rlata=9.5e-2/2;                     #m Radio lata
lcel=0.16;                          #m largo del celu
acel=0.08                           #m ancho del celuz_zero
ltor=10.5e-2;                       #Largo tornillo
ltue=1e-2                           #Largo tuerca
di=Rlata-ltue;                      #distiantia inicio tornillo
df=di+ltor;                         #distiantia final tornillo
Dres=7e-2;                          #Diametro resorte


##################################################    
# mass Data  (in kg)
################################################
mtapa=9e-3;                         #kg
mlata=31.4e-3;                      #kg
mcel=228e-3;                        #kg
mtor=56.29e-3;                      #kg masa tornillo
mtue=6.74e-3;                       #kg masa tuerca
mlin=(mtor+mtue*2)/ltor;            #masa lineal tuerca (hypothesis: seccion constante)
m=mcel+mlata+mtor*2+mtue*4;         #total mass


##################################################    
# moments of inertia
################################################
Icaja=mtapa*2./5*Rlata**2+(mlata-mtapa)*Rlata**2    #box inertia
Icel=(acel**2+lcel**2)/12*mcel                      #Smartphone inerta
Itor=mlin*(df**3-di**3)/3                           #Bolts inertia

I=2*Itor+Icel+Icaja                                 #total inertia


##################################################    
# coupling term
################################################
epsilon=omegaB*omega*2*(m*I)**0.5

##################################################    
# Initial conditions
################################################
z_zero=1.40-np.array([0.58,0.58])
theta_zero=np.array([0,-epsilon/2/I/omega**2*z_zero[1]])

plt.close("all")
##################################################    
# reading and ploting normal modes
################################################
Nfile=2
[z_acc1, theta_acc1, z_vel, theta_vel, z, theta, theta_accTF, z_accTF, Freq, Time1,Nf]=ReadData(Nfile)
Nfile=3
[z_acc2, theta_acc2, z_vel, theta_vel, z, theta, theta_accTF, z_accTF, Freq, Time2,Nf]=ReadData(Nfile)  

[ax10,ax20,ax30,fig0]=plotyyDbl(Time1,z_acc1,theta_acc1,Time2,z_acc2,theta_acc2,ylabel1='$\ddot{z}$ (m/s$^2$)',ylabel2='$\ddot{\Theta}$ (rad/s$^2$)',ylabel3='$\ddot{z}$ (m/s$^2$)',ylabel4='$\ddot{\Theta}$ (rad/s$^2$)')    
ax10[0].set_ylim([-1, 1])
ax10[1].set_ylim([-1, 1]) 
ax10[0].set_xlim([-5, 40])
ax10[1].set_xlim([-5, 40]) 
ax20.set_ylim([-15, 15])
ax30.set_ylim([-15, 15]) 
ax10[0].text(-12,1, 'a)', fontsize=12)
ax10[1].text(-12,1.2, 'b)', fontsize=12)
    

##################################################    
# reading beating 1
################################################
Nfile=1
[z_acc, theta_acc, z_vel, theta_vel, z, theta, theta_accTF, z_accTF, Freq, Time,Nf]=ReadData(Nfile)
dt=np.mean(np.diff(Time))
fs=1/dt
Nt=Time.shape[0]

##################################################    
# numerical simulation
################################################

# numerical time vector
TimeNum=np.linspace(0,Time[-1]-Time[0],Nt)
# Initial conditions:
y0 = [z_zero[Nfile], 0, theta_zero[Nfile], 0] 
# attenuation coef set arbitrarily
alpha=0.03/m            
zeta=alpha/2/omega      # azeta coef 
# Do the numerical integration of the equations of motion
y = odeint(derivAtt, y0, TimeNum, args=(omega, epsilon, m, I,alpha))
 # Unpack z and theta as a function of time
z_num,z_velNum, theta_num,theta_velNum= y[:,0], y[:,1], y[:,2], y[:,3]
#compute the accelerations
z_accNum=np.gradient(z_velNum,TimeNum);
theta_accNum=np.gradient(theta_velNum,TimeNum);

#fft parameter
Npad=10;
NTF=800;
Nf=Npad*z_acc[Nh:].shape[0]
z_accTF=np.fft.fft(z_acc[Nh:NTF],axis=0,n=Nf)
#fft computation
Freq=np.fft.fftfreq(Nf,dt)   
theta_accTF=np.fft.fft(theta_acc[Nh:NTF],axis=0,n=Nf)
theta_velTF=np.fft.fft(theta_vel[Nh:NTF],axis=0,n=Nf)
z_accNumTF=np.fft.fft(z_accNum,axis=0,n=Nf)
theta_velNumTF=np.fft.fft(theta_velNum[Nh:NTF],axis=0,n=Nf)


##################################################    
# plot translation in time and frequency 
################################################
[ax1,fig]=plotDbl(TimeNum,z_accNum,z_acc,Freq[1:int(-Nf/2)],np.abs(z_accNumTF[1:int(-Nf/2)])**2,np.abs(z_accTF[1:int(-Nf/2)])**2,xlabel1='Time (s)',xlabel2='Frequency (Hz)',ylabel1='$\ddot{z}$  (m/s$^2$)',ylabel2='Intensity (a.u.)',t2=Time,Dt2=Freq[1:int(-Nf/2)],col1='k',col2='k',linestyle1=':',linestyle2='-')    
ax1[0].set_ylim([-np.max(np.abs(z_accNum)),np.max(np.abs(z_accNum))])
ax1[0].set_xlim([-3,7/(f2-f1)])
fmin=0
fmax=1
ax1[1].set_xlim([fmin,fmax])
ax1[1].set_yticks([])
ax1[1].set_yticks([])
ax1[0].text(-12,np.max(np.abs(z_acc)), 'a)', fontsize=12)
ax1[1].text(-0.15,np.max(np.abs(z_accTF**2)), 'b)', fontsize=12)
#plt.close("all")

    
##################################################    
# compute the amplitude frequency 
################################################
peaksNum, propertiesNum=signal.find_peaks(np.abs(z_accNumTF[1:int(-Nf/2)])**2, height=20000)
HeightsNum=propertiesNum["peak_heights"]
np.power(HeightsNum[0]/HeightsNum[1],0.5)
(Freq[peaksNum[0]]*2*np.pi-omega1)/omega1
(Freq[peaksNum[1]]*2*np.pi-omega2)/omega2
omega1**2/omega2**2
peaks,  properties=signal.find_peaks(np.abs(z_accTF[1:int(-Nf/2)])**2, height=80000)
Heights=properties["peak_heights"]
(Freq[peaks[0]]*2*np.pi-omega1)/omega1
(Freq[peaks[1]]*2*np.pi-omega2)/omega2
np.power(Heights[0]/Heights[1],0.5)
omega1**2/omega2**2
    
    
##################################################    
# reading beating 2
################################################
Nfile=1
[z_acc, theta_acc, z_vel, theta_vel, z, theta, theta_accTF, z_accTF, Freq, Time,Nf]=ReadData(Nfile)
dt=np.mean(np.diff(Time))
fs=1/dt
Nt=Time.shape[0]

##################################################    
# numerical simulation
################################################
TimeNum=np.linspace(0,Time[-1]-Time[0],Nt)
# Initial conditions: theta=2pi, z=zdot=thetadot=0
y0 = [z_zero[Nfile], 0, theta_zero[Nfile], 0] 
# Do the numerical integration of the equations of motion
y = odeint(derivAtt, y0, TimeNum, args=(omega, epsilon, m, I,alpha))
# Unpack z and theta as a function of time
z_num,z_velNum, theta_num,theta_velNum= y[:,0], y[:,1], y[:,2], y[:,3]
#compute the accelerations
z_accNum=np.gradient(z_velNum,TimeNum);
theta_accNum=np.gradient(theta_velNum,TimeNum);


Npad=10
NTF=800;
Nf=Npad*z_acc[Nh:].shape[0]
#fft computation
Freq=np.fft.fftfreq(Nf,dt)  
z_accTF=np.fft.fft(z_acc[0:NTF],axis=0,n=Nf)
theta_accTF=np.fft.fft(theta_acc[0:NTF],axis=0,n=Nf)
z_accNumTF=np.fft.fft(z_accNum,axis=0,n=Nf)
theta_accNumTF=np.fft.fft(theta_accNum[0:NTF],axis=0,n=Nf)


  


##################################################    
# plot translation and rotation  in time 
################################################
# beating time
TB=1/(omegaB/np.pi/2)
[ax1,fig]=plotDbl(TimeNum*omegaB/np.pi/2,z_accNum,z_acc,TimeNum*omegaB/np.pi/2,theta_accNum,theta_acc,xlabel1='Time ($T_B$)',xlabel2='Time ($T_B$)',ylabel1='$\ddot{z}$  (m/s$^2$)',ylabel2='$\ddot{\Theta}$  (rad/s$^2$)',col2='r',col1='k',linestyle1=':',linestyle2='-',t2=Time*omegaB/np.pi/2,Dt2=Time*omegaB/np.pi/2)    

ax1[0].set_ylim([-np.max(np.abs(z_accNum)),np.max(np.abs(z_accNum))])
ax1[1].set_ylim([-np.max(np.abs(theta_accNum)),np.max(np.abs(theta_accNum))])
ax1[0].set_xlim([-0.5,7])
ax1[1].set_xlim([-0.5,7])
ax1[0].set_yticks([-5,0,5])
ax1[0].grid(True)
ax1[1].set_yticks([-50,0,50])
ax1[1].grid(True)

ax1[0].text(-1.3,np.max(np.abs(z_acc)), 'a)', fontsize=12)
ax1[1].text(-1.3,np.max(np.abs(theta_acc)), 'b)', fontsize=12)

##################################################    
# plot translation and rotation in frequency
################################################
[ax1,fig]=plotDbl(Freq[1:int(-Nf/2)],np.abs(z_accNumTF[1:int(-Nf/2)])**2,np.abs(z_accTF[1:int(-Nf/2)])**2,Freq[1:int(-Nf/2)],np.abs(theta_accNumTF[1:int(-Nf/2)])**2,np.abs(theta_accTF[1:int(-Nf/2)])**2,xlabel1='Frequency (Hz)',xlabel2='Frequency (Hz)',ylabel1='$\ddot{z}$- Intensity (a.u.)',ylabel2='$\ddot{\Theta}$- Intensity (a.u.)',col2='r',col1='k',linestyle1=':',linestyle2='-',t2=Freq[1:int(-Nf/2)],Dt2=Freq[1:int(-Nf/2)])    
fmin=0
fmax=1
ax1[1].set_xlim([fmin,fmax])
ax1[1].set_yticks([])
ax1[0].set_xlim([fmin,fmax])
ax1[0].set_yticks([])

ax1[0].text(-0.15,np.max(np.abs(z_accTF**2)), 'a)', fontsize=12)
ax1[1].text(-0.15,np.max(np.abs(theta_accTF**2)), 'b)', fontsize=12)

    
##################################################    
# compute the amplitude frequency ratio
################################################    
peaksNum, propertiesNum=signal.find_peaks(np.abs(z_accNumTF[1:int(-Nf/2)])**2, height=100000)
HeightsNum=propertiesNum["peak_heights"]
HeightsNum[0]/HeightsNum[1]
peaks,  properties=signal.find_peaks(np.abs(z_accTF[1:int(-Nf/2)])**2, height=100000)
Heights=properties["peak_heights"]
np.power(Heights[0]/Heights[1],0.5)


peaks,  properties=signal.find_peaks(np.abs(theta_accTF[1:int(-Nf/2)])**2, height=100000)
Heights=properties["peak_heights"]
np.power(Heights[0]/Heights[1],0.5)


##################################################    
# representation for cyilindrical coordinate
################################################
filename=filenameAll[Nfile]
[Time,Data,headings] =readcsv(filename+'.csv')

Time=Time[int(Ini[Nfile]-Nh):int(Fin[Nfile])]-Time[int(Ini[Nfile])]
Data= Data[int(Ini[Nfile]-Nh):int(Fin[Nfile]),:]
x_acc=Data[:,0]
y_acc=Data[:,1]
dt=np.mean(np.diff(Time))
fs=1/dt
Nt=Time.shape[0]

# optimization for distance estimation
banana = lambda x: sum((y_acc[50:500]+x*theta_vel[50:500]**2)**2)
ry = scipy.optimize.fmin(func=banana, x0=0.035)


fig,ax1=plt.subplots(1,1)
ax1.plot(Time,-ry*theta_vel**2,color='r')
ax1.plot(Time,y_acc,color='k')
ax1.set_xlim([-3,7/(f2-f1)])
ax1.set_xlabel('Time')
ax1.set_ylabel('$a_r$ (m/s$^2$)')
ax1.legend(['$r\dot{\Theta}^2$', '$\ddot{y}$'])


