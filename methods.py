import torch
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian

def SEISmodel(theta, L, S, E, I):
    alpha=0.2
    mu=0.01
    beta = theta[0]
    gamma = theta[1]
    Ks = theta[2]
    Ke= theta[2]
    Ki = theta[3]
    for i in range (13):
        dSdt = -Ks*L@S-beta * E * S - gamma * I * S # dS/dt
        dEdt = -Ke*L@E+beta * E * S + gamma * I * S - alpha * E  # dE/dt
        dIdt = -Ki*L@I+alpha * E - mu * I  # dI/dt

    return dSdt, dEdt, dIdt


def integrateSEIS(theta, S0, E0, I0, dt, nt):
    # vectors to save the results over time
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
    Sout = torch.zeros(13,nt + 1)
    Sout[:,0] = S0
    Eout = torch.zeros(13,nt + 1)
    Eout[:,0] = E0
    Iout = torch.zeros(13,nt + 1)
    Iout[:,0] = I0

    S = S0
    E = E0
    I = I0
    for i in range(nt):
            dSdt, dEdt, dIdt = SEISmodel(theta, L, S, E, I)
            S += dt * dSdt
            E += dt * dEdt
            I += dt * dIdt

            Sout[:,i + 1] = S
            Eout[:,i + 1] = E
            Iout[:,i + 1] = I

    return Sout, Eout, Iout

def loss(Sobs,Eobs,Iobs, theta, S0, E0, I0, dt, nt):
    Scomp, Ecomp, Icomp=integrateSEIS(theta, S0, E0,I0,dt,nt)
    phi=torch.sum((Icomp-Iobs)**2)

    return phi
#TODO:edit functions for this case

def gradient(theta):
    phi=loss(theta)
    phi.backward()
    grady=theta.grad
    return grady

def Steepest_Descent(theta, mu):
    g=gradient(theta)
    theta=theta-mu*g
    return theta
#TODO:set up date checking
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
        idx = pd.date_range(start='01-31-2020',end='04-19-2020',freq='D')
        data=data.reindex(idx)
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