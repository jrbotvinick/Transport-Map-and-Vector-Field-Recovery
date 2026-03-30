import numpy as np
import matplotlib.pyplot as plt
import torch
from geomloss import SamplesLoss
from torch import optim
import torch.nn as nn
import math
from scipy import stats
import pickle 

import random

np.random.seed(84251)
torch.manual_seed(1231)

for iii in range(10):
    
        
    def lorenz(xyz, *, s=10, r=28, b=2.667):
        x, y, z = xyz
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])
    
    def lorenz_vec(xyz, *, s=10, r=28, b=2.667):
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot]).T
    
    
    dt = 0.01
    N = int(1e5)
    N_evolve = 10
    num_densities = 8
    N_iterations = int(1e4)
    batch_size = 200
    permute = True #True or false
    
    
    init = [np.random.uniform(-15,15),np.random.uniform(-15,15),np.random.uniform(20,40)]
    X = init+np.random.normal(0,1,(N,3))*np.random.uniform(3,7)
    Xs = []
    def evolve(rho):
        X = rho.copy()
        for i in range(N_evolve):
            X = lorenz_vec(X)*dt+X
        return X
    
    def evolve_NN(X,f):
        Y = X.clone()
        for i in range(N_evolve):    
            Y = f(Y)*dt+Y
        return Y
        
    for i in range(num_densities):
        if permute == True:
            Xs.append(np.random.permutation(X))
        else:
            Xs.append(X)
    
        X = evolve(X)
    
    ixs = [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3] ]
    fig,ax = plt.subplots(2,4,figsize = (10,5),dpi = 300)
    for i in range(8):
        ax[ixs[i][0],ixs[i][1]].set_title('t = {}'.format(i*N_evolve*dt))
        ax[ixs[i][0],ixs[i][1]].hist2d(Xs[i][:,0],Xs[i][:,2],range = [[-25,25],[0,50]],bins = 100,vmin = 0,vmax= 100)
    plt.subplots_adjust(wspace = 0.5,hspace = 0.5)
    plt.show()
    
    
    
    net = nn.Sequential(
        nn.Linear(3, 100),
        nn.Tanh(),
        nn.Linear(100,100),
        nn.Tanh(),
        nn.Linear(100,100),
        nn.Tanh(),
        nn.Linear(100,3))
    
    loss = SamplesLoss(loss="energy")
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    
    
    M = np.max(np.array(Xs))
    Xs = Xs/M
    
    for i in range(N_iterations):
        ixs = np.random.choice(range(N),batch_size)
    
        optimizer.zero_grad()
        
        YY, ZZ = torch.zeros((num_densities-1,batch_size,3),dtype = torch.float),torch.zeros((num_densities-1,batch_size,3),dtype = torch.float)
        for j in range(num_densities-1):
            Y = evolve_NN(torch.tensor(Xs[j][ixs,:],dtype =torch.float),net)
            Z = torch.tensor(Xs[j+1][ixs,:],dtype =torch.float)
            YY[j] = Y
            ZZ[j] = Z
        L = loss(YY,ZZ)
        L = L.sum()
        L.backward()
        optimizer.step()
        
        if i%100==0:
            print('Iteration: ',i, ' Loss: ', L.detach().numpy())
            
            x = torch.tensor(np.array([0.1,0.1,0.1]),dtype = torch.float)
            xs = []
            for k in range(int(1e4)):
                xs.append(x.detach().numpy())
                x = x + dt*net(x)
            xs = np.array(xs)
            
            
            y = np.array([0.1,0.1,0.1])*M
            ys = []
            for k in range(int(1e4)):
                ys.append(y)
                y = y + dt*lorenz(y)
            ys = np.array(ys)
    
            
            fig = plt.figure(dpi = 300)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(ys[:,0],ys[:,1],ys[:,2],linewidth = 1)
    
            ax.plot(xs[:,0]*M,xs[:,1]*M,xs[:,2]*M,linewidth = 1)
            plt.show()
            
           
                
            
    
            
            
            
        
    torch.save(net.state_dict(), "lorenz_pushforward_{}.pth".format(iii))
    
    with open("lorenz_experiment_{}.p".format(iii), "wb") as f:
        pickle.dump([dt,N_evolve,Xs,M], f)
    
        
        






