import torch
import torch.nn as nn
class SEIModel(nn.Module):
    def _init_(self):
        super._init_()
        self.beta=nn.Parameter(torch.tensor(0.3,requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.K = nn.Parameter(torch.tensor(0.2, requires_grad=True))
        self.Ki = nn.Parameter(torch.tensor(0.01, requires_grad=True))

    def forward(self):
        alpha = 0.2
        mu = 0.01


    def SEISmodel(beta, gamma, K, Ki, L, S, E, I):
        alpha = 0.2
        mu = 0.01
        Ks = K
        Ke = K
        for i in range(13):
            dSdt = -Ks * L @ S - beta * E * S - gamma * I * S  # dS/dt
            dEdt = -Ke * L @ E + beta * E * S + gamma * I * S - alpha * E  # dE/dt
            dIdt = -Ki * L @ I + alpha * E - mu * I  # dI/dt

        return dSdt, dEdt, dIdt

    def integrateSEIS(beta, gamma, K, Ki, S0, E0, I0, dt, nt):
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
            dSdt, dEdt, dIdt = SEISmodel(beta, gamma, K, Ki, L, S, E, I)
            Sout[:, i + 1] = Sout[:, i] + dt * dSdt
            Eout[:, i + 1] = Eout[:, i] + dt * dEdt
            Iout[:, i + 1] = Iout[:, i] + dt * dIdt

            # Sout[:,i + 1] = S
            # Eout[:,i + 1] = E
            # Iout[:,i + 1] = I

        return Sout, Eout, Iout