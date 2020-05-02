import torch
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian

def SEISmodel(theta, L, S, E, I):
    alpha=0.2
    mu=0.01
    beta=theta[0]
    gamma=theta[1]
    K=theta[2]
    Ks = K
    Ke= K
    Ki=theta[3]
    for i in range (13):
        dSdt = -theta[2]*L@S-theta[0] * E * S - theta[1] * I * S # dS/dt
        dEdt = -theta[2]*L@E+theta[0] * E * S + theta[1] * I * S - alpha * E  # dE/dt
        dIdt = -theta[3]*L@I+alpha * E - mu * I  # dI/dt

    return dSdt, dEdt, dIdt

def integrateSEIS(theta, S0, E0, I0, dt, nt):
    A = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
                  [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
                  [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]])
    L = torch.from_numpy(laplacian(A))
    Sout = torch.ones(13,nt,requires_grad=False)
    Sout[:,0] = S0
    Eout = torch.zeros(13,nt,requires_grad=False)
    Eout[:,0] = E0
    Iout = torch.zeros(13,nt,requires_grad=False)
    Iout[:,0] = I0

    S = S0
    E = E0
    I = I0
    for i in range(nt-1):
            dSdt, dEdt, dIdt = SEISmodel(theta, L, S, E, I)
            Sout[:,i+1] = Sout[:,i]+dt * dSdt
            Eout[:,i+1]=Eout[:,i]+dt * dEdt
            Iout[:,i+1] = Iout[:,i]+dt * dIdt

            #Sout[:,i + 1] = S
            #Eout[:,i + 1] = E
            #Iout[:,i + 1] = I

    return Sout, Eout, Iout

def loss(Iobs,theta, S0, E0, I0,dt,nt):
    Scomp, Ecomp, Icomp=integrateSEIS(theta, S0, E0,I0,dt,nt)
    phi=torch.sum((Icomp-Iobs)**2)
    return phi

def gradient(theta):
    phi=loss(theta)
    phi.backward()
    grady=theta.grad
    return grady

def Steepest_Descent(theta, mu):
    g=gradient(theta)
    theta=theta-mu*g
    return theta

def import_csv(enddate):
    pruid=[48,59,46,13,10,61,12,62,35,11,24,47,60]
    df=pd.read_csv('covid19.csv')
    df=pd.DataFrame(df,columns =['pruid','date','numconf'])
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    torch_data = torch.zeros(13,80)
    for i in range (13):
        data=df[df.pruid.eq(pruid[i])]
        data.set_index('date', inplace=True)
        data.sort_index(inplace=True)
        idx = pd.date_range(start='01-31-2020',end=enddate,freq='D')
        data=data.reindex(idx)
        if data['numconf']['01-31-2020']!=data['numconf']['01-31-2020']:
            data['numconf']['01-31-2020']=0
        data=data.fillna(method='ffill')
        t = torch.tensor(data['numconf'].values)
        for j in range (80):
            torch_data[i,j]=t[j]
    return torch_data

def switch(argument):
    switcher = {
        0:"48",
        1:"59",
        2:"46",
        3:"13",
        4:"10",
        5:"61",
        6:"12",
        7:"62",
        8:"35",
        9:"11",
        10:"24",
        11:"47",
        12:"60"
    }
    output=switcher.get(argument,13)
    return output