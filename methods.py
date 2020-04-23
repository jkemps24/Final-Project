import torch

def SEISmodel(theta, L, S, E, I):
    alpha = theta[0]
    beta = theta[1]
    gamma = theta[2]
    mu = theta[3]
    Ks = theta[4]
    Ke= theta[4]
    Ki = theta[5]
    for i in range (13):
        dSdt = -Ks*L@S-beta * E * S - gamma * I * S # dS/dt
        dEdt = -Ke*L@E+beta * E * S + gamma * I * S - alpha * E  # dE/dt
        dIdt = -Ki*L@I+alpha * E - mu * I  # dI/dt

    return dSdt, dEdt, dIdt


def integrateSEIS(theta, L, S0, E0, I0, dt, nt):
    # vectors to save the results over time
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

def loss(Sobs,Eobs,Iobs,Scomp,Ecomp,Icomp):
    phi=torch.sum((Scomp-Sobs)**2)+torch.sum((Ecomp-Eobs)**2)+torch.sum((Icomp-Iobs)**2)

    return phi
#TODO:edit minmuim function for this case
def minimuim(theta,mu, nu):
    d=1
    while d > nu:
        f=function(theta)
        theta=Steepest_Descent(theta,mu)
        d=abs(f-function(theta))
        print(theta)

    return theta

def gradient(theta):
    y=function(theta)
    y.backward()
    grady=theta.grad
    return grady

def Steepest_Descent(theta, mu):
    g=gradient(theta)
    theta=theta-mu*g
    return theta
