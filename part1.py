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
alpha=torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
beta=torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
gamma=torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
mu=torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
Ks=torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
Ke=torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
Ki=torch.tensor([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
theta=torch.tensor([alpha,beta,gamma,mu,Ks,Ke,Ki], requires_grad=True)
mu1=0.001
nu=0.00001
#𝑃={′𝐴𝐵′,′𝐵𝐶′,′𝑀𝐵′,′𝑁𝐵′,′𝑁𝐿′,′𝑁𝑇′,′𝑁𝑆′,′𝑁𝑈′,′𝑂𝑁′,′𝑃𝐸′,′𝑄𝐶′,′𝑆𝐾′,′𝑌𝑇′}
S0=torch.FloatTensor([4345737,5020302,1360396,772094,523790,44598,965382,38787,14446515,154748,8433301,1168423,40369])
E0=torch.zeros(13)
I0=torch.FloatTensor([0,1,0,0,0,0,0,0,3,0,0,0,0])
dt=1.0
nt=100
#minimuim(theta,mu1,nu)
S,E,I=integrateSEIS(theta,L,S0,E0,I0,dt,nt)

t=torch.arange(nt+1)*dt
for i in range (13):
    plt.plot(t,S[i],t,E[i],t,I[i])

plt.show()