import numpy as np
import matplotlib.pyplot as plt
import torch
from geomloss import SamplesLoss
from torch import optim
import torch.nn as nn
import math
from scipy import stats
import pickle 



ms = [1,2,3,4,5,6,7]
centers = np.array([np.random.uniform(-1,1,8),np.random.uniform(-1,1,8)]).T
sigmas = [np.random.uniform(0.75,1.25) for i in range(8)]

plt.scatter(centers[:,0],centers[:,1])
plt.show()
for m in ms:
    for ll in range(10):
        N = 100
        Niters = int(2e4)
        batch = 250
        a = 4
        def V(X):
            x,y = X[:,0],X[:,1]
            xdot,ydot = y, -np.sin(4*math.pi*x)
            return np.array([xdot, ydot]).T
            
        def Vt(X):
            x = X[:, 0]
            y = X[:, 1]
            xdot,ydot = y, - torch.sin(a*torch.pi*x)
            return torch.stack((xdot, ydot), dim=1)
        
        
        x,y = np.linspace(-1,1,N),np.linspace(-1,1,N)
        
        XX = np.zeros((N,N,2))
        XX[:,:,0], XX[:,:,1] = np.meshgrid(x,y)
        XX = XX.reshape(N**2,2)
        v = V(XX)
        
        plt.streamplot(XX[:,0].reshape(N,N),XX[:,1].reshape(N,N),v[:,0].reshape(N,N),v[:,1].reshape(N,N),density = 2)
        plt.show()
        
        
        
        
        def eval_gauss(x,center,sigma):
            return (1/(2*math.pi*sigma**2) ) * np.exp(- ((x[:,0]-center[0])**2+(x[:,1]-center[1])**2)/(2*sigma**2))
        
        def eval_gausst(x,center,sigma):
            return ((1/(2*torch.pi*sigma**2) ) * torch.exp(- ((x[:,0]-center[0])**2+(x[:,1]-center[1])**2)/(2*sigma**2))).unsqueeze(1)
        
        
        gs = []
        for j in range(m):
            gs.append(eval_gauss(XX,centers[j],sigmas[j]))
            
        gs = np.array(gs)
        g = np.max(gs,axis = 0)
        
        
        plt.contourf(x,y,g.reshape(N,N),levels = 30)
        plt.scatter(centers[:m,0],centers[:m,1],color = 'k',marker = 'x')
        plt.show()
    
        net = nn.Sequential(
             nn.Linear(2, 50),
             nn.Tanh(),
             nn.Linear(50,50),
             nn.Tanh(),
             nn.Linear(50,50),
             nn.Tanh(),
             nn.Linear(50,2))
        
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        
        
        def divergence(v, x):
        
            dvx = torch.autograd.grad(
                outputs = v[:,0],
                inputs = x,
                grad_outputs=torch.ones_like(v[:,0]),
                create_graph=True,
                retain_graph=True
            )[0][:,0]
        
            dvy = torch.autograd.grad(
                outputs = v[:,1],
                inputs = x,
                grad_outputs=torch.ones_like(v[:,1]),
                create_graph=True,
                retain_graph=True
            )[0][:,1]
        
            return dvx + dvy
        
        
        for i in range(Niters):
            optimizer.zero_grad()
            
            Y = torch.tensor(np.array([np.random.uniform(-1,1,batch),np.random.uniform(-1,1,batch)]).T,dtype = torch.float,requires_grad = True)
            V_true = Vt(Y)
            V_net = net(Y)
            
            L = torch.tensor(np.array([0]),dtype = torch.float)[0]
            
            for j in range(m):
                y1 = V_true*eval_gausst(Y,centers[j],sigmas[j])
                y2 = V_net*eval_gausst(Y,centers[j],sigmas[j])
                d1 = divergence(y1,Y)
                d2 = divergence(y2,Y)
                L+= torch.mean((d1-d2)**2)
                
            
            L.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print('Iteration: ', i, '| Loss: ', L)
                
            if i % 1000 == 0:
                ZZ = torch.tensor(XX,dtype = torch.float)
                vv = net(ZZ).detach().numpy()
                plt.streamplot(XX[:,0].reshape(N,N),XX[:,1].reshape(N,N),vv[:,0].reshape(N,N),vv[:,1].reshape(N,N),density = 2)
                plt.show()
                
            ZZ = torch.tensor(XX,dtype = torch.float)
            vv = net(ZZ).detach().numpy()
        with open("divergence_experiment_{}_{}.p".format(m,ll), "wb") as f:
            pickle.dump([v,vv,XX,centers,sigmas], f)
            
        
                
                
                
                
                
        
                
            
            
        
    
