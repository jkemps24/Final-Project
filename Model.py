import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.csgraph import laplacian

class SEIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta=nn.Parameter(torch.tensor(0.1,requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.K = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.Ki = nn.Parameter(torch.tensor(0.1, requires_grad=True))



    def SEISmodel(self, L, S, E, I):
        alpha = 0.2
        mu = 0.01
        for i in range(13):
            dSdt = -self.K * L @ S - self.beta * E * S - self.gamma * I * S  # dS/dt
            dEdt = -self.K * L @ E + self.beta * E * S + self.gamma * I * S - alpha * E  # dE/dt
            dIdt = -self.Ki * L @ I + alpha * E - mu * I  # dI/dt

        return dSdt, dEdt, dIdt

    def forward(self, S0, E0, I0, dt, nt):
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
        Sout = torch.ones(13, nt, requires_grad=False)
        Sout[:, 0] = S0
        Eout = torch.zeros(13, nt, requires_grad=False)
        Eout[:, 0] = E0
        Iout = torch.zeros(13, nt, requires_grad=False)
        Iout[:, 0] = I0

        S = S0
        E = E0
        I = I0
        for i in range(nt - 1):
            dSdt, dEdt, dIdt = self.SEISmodel(L, S, E, I)
            Sout[:, i + 1] = Sout[:, i] + dt * dSdt
            Eout[:, i + 1] = Eout[:, i] + dt * dEdt
            Iout[:, i + 1] = Iout[:, i] + dt * dIdt

        out=torch.zeros(2,13,nt)
        out[0]=Sout
        out[1]=Iout

        return out