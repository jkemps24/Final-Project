
import torch
import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
from methods import integrateSEIS,minimuim
A = np.array([[1,     1,     1,     0,     0,     0,     0,     0,     1,     0,     1,     1,     1],
     [1,     1,     1,     0,     0,     1,     0,     0,     1,     0,     1,     1,     0],
     [1,     1,     1,     0,     0,     1,     0,     1,     1,     0,     1,     1,     0],
     [0,     0,     0,     1,     1,     0,     0,     0,     1,     1,     1,     0,     0],
     [0,     0,     0,     1,     1,     0,     1,     0,     1,     1,     1,     0,     0],
     [0,     1,     1,     0,     0,     1,     0,     1,     1,     0,     1,     1,     1],
     [0,     0,     0,     0,     1,     0,     1,     0,     1,     1,     1,     0,     0],
     [0,     0,     1,     0,     0,     1,     0,     1,     1,     0,     1,     1,     1],
     [1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
     [0,     0,     0,     1,     1,     0,     1,     0,     1,     1,     1,     0,     0],
     [1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
     [1,     1,     1,     0,     0,     1,     0,     1,     1,     0,     1,     1,     0],
     [1,     0,     0,     0,     0,     1,     0,     1,     1,     0,     1,     0,     1]])
L=laplacian(A)
L=torch.from_numpy(L)
# 𝑃={′𝐴𝐵′,′𝐵𝐶′,′𝑀𝐵′,′𝑁𝐵′,′𝑁𝐿′,′𝑁𝑇′,′𝑁𝑆′,′𝑁𝑈′,′𝑂𝑁′,′𝑃𝐸′,′𝑄𝐶′,′𝑆𝐾′,′𝑌𝑇′}
fill=torch.ones(13)
# alpha and mu are disease dependant
alpha=torch.tensor([0.2])
beta=fill.fill_(0.3)
gamma=fill.fill_(0.01)
mu=torch.tensor([0.01])
# K=Ks=Ke as susceptible pop is just as mobile as effected
K=fill.fill_(0.2)
Ki=fill.fill_(0.01)
theta=torch.cat([alpha,beta,gamma,mu,K,Ki])

S0=torch.FloatTensor([4345737.0,5020302.0,1360396.0,772094.0,523790.0,44598.0,965382.0,38787.0,14446515.0,154748.0,8433301.0,1168423.0,40369.0])
E0=torch.FloatTensor([1,5,1,0,0,0,0,0,10,0,3,1,0])
I0=torch.FloatTensor([0,1,0,0,0,0,0,0,3,0,0,0,0])
dt=1.0
nt=100
#minimuim(theta,mu1,nu)
S,E,I=integrateSEIS(theta,L,S0,E0,I0,dt,nt)

t=torch.arange(nt+1)*dt
for i in range (13):
     plt.figure(i)
     plt.plot(t,E[i],t,I[i])
plt.show()