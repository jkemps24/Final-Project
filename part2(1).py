import torch
import torch.optim as optim
from methods import loss, Steepest_Descent,import_csv

# 𝑃={′𝐴𝐵′,′𝐵𝐶′,′𝑀𝐵′,′𝑁𝐵′,′𝑁𝐿′,′𝑁𝑇′,′𝑁𝑆′,′𝑁𝑈′,′𝑂𝑁′,′𝑃𝐸′,′𝑄𝐶′,′𝑆𝐾′,′𝑌𝑇′}
# alpha and mu are disease dependant
beta=0.3
gamma=0.01
# K=Ks=Ke as susceptible pop is just as mobile as effected
K=0.2
Ki=0.01
#with torch.autograd.set_detect_anomaly(True):
theta=torch.tensor([beta,gamma,K,Ki],requires_grad=True)

S0=torch.FloatTensor([4345737.0,5020302.0,1360396.0,772094.0,523790.0,44598.0,965382.0,38787.0,14446515.0,154748.0,8433301.0,1168423.0,40369.0])
E0=torch.FloatTensor([1,5,1,0,0,0,0,0,10,0,3,1,0])
I0=torch.FloatTensor([0,1,0,0,0,0,0,0,3,0,0,0,0])
Iobs=import_csv('19-04-2020')
dt=1.0
nt=Iobs[0].shape
nt=nt[0]
nu=0.0001
mu=0.001
d=1
#optimizer = optim.SGD([theta],lr=mu)

while d > nu:
    phi=loss(Iobs, theta, S0, E0, I0,dt,nt)
    phi.backward()
    #optimizer.step()
    grady=theta.grad
    with torch.no_grad():
        theta -=mu*grady
    d=abs(phi-loss(Iobs, theta, S0, E0, I0,dt,nt))
    theta.grad.zero_()
    #optimizer.zero_grad()
    print (phi)
    print(theta)