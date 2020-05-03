
import torch
import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
from methods import integrateSEIS

# 𝑃={′𝐴𝐵′,′𝐵𝐶′,′𝑀𝐵′,′𝑁𝐵′,′𝑁𝐿′,′𝑁𝑇′,′𝑁𝑆′,′𝑁𝑈′,′𝑂𝑁′,′𝑃𝐸′,′𝑄𝐶′,′𝑆𝐾′,′𝑌𝑇′}
# alpha and mu are disease dependant
beta=0.3
gamma=0.01
# K=Ks=Ke as susceptible pop is just as mobile as effected
K=0.2
Ki=0.01
theta=torch.tensor([beta,gamma,K,Ki])

S0=torch.FloatTensor([4345737,5020302,1360396,772094,523790,44598,965382,38787,14446515,154748,8433301,1168423,40369])
E0=torch.FloatTensor([1,5,1,0,0,0,0,0,10,0,3,1,0])
I0=torch.FloatTensor([0,1,0,0,0,0,0,0,3,0,0,0,0])
dt=1.0
nt=100
#minimuim(theta,mu1,nu)
S,E,I=integrateSEIS(theta,S0,E0,I0,dt,nt)

t=torch.arange(nt+1)*dt
for i in range (13):
     plt.figure(i)
     plt.plot(t,E[i],t,I[i])
plt.show()