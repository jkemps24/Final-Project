import torch
from methods import loss, Steepest_Descent

# 𝑃={′𝐴𝐵′,′𝐵𝐶′,′𝑀𝐵′,′𝑁𝐵′,′𝑁𝐿′,′𝑁𝑇′,′𝑁𝑆′,′𝑁𝑈′,′𝑂𝑁′,′𝑃𝐸′,′𝑄𝐶′,′𝑆𝐾′,′𝑌𝑇′}
# alpha and mu are disease dependant
beta=torch.ones(13)*0.3
gamma=torch.ones(13)*0.01
# K=Ks=Ke as susceptible pop is just as mobile as effected
K=torch.ones(13)*0.2
Ki=torch.ones(13)*0.01
theta=torch.cat([beta,gamma,K,Ki])

S0=torch.FloatTensor([4345737.0,5020302.0,1360396.0,772094.0,523790.0,44598.0,965382.0,38787.0,14446515.0,154748.0,8433301.0,1168423.0,40369.0])
E0=torch.FloatTensor([1,5,1,0,0,0,0,0,10,0,3,1,0])
I0=torch.FloatTensor([0,1,0,0,0,0,0,0,3,0,0,0,0])
dt=1.0
nt=100
#Iobs=
#Sobs=
nu=0.0001
mu=0.001
d=1

for i in range (13):
    theta=torch.tensor([beta[i],gamma[i],K[i],Ki[i]])
    while d > nu:
        phi=loss(Sobs[i], Eobs[i], Iobs[i], theta[i], nt, dt, S0[i], E0[i], I0[i])
        phi.backward()
        grady=theta.grady
        theta -=mu*grady
        d=abs(phi-loss(theta))
    print(theta)