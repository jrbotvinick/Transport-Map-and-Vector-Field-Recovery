import numpy as np
import matplotlib.pyplot as plt
import torch
from geomloss import SamplesLoss
from torch import optim
import torch.nn as nn
import math
from scipy import stats
from torch.distributions import VonMises
import pickle




np.random.seed(541293)
torch.manual_seed(221259)

for iii in range(10):
    
    d = 3
    def ft(x):
        return torch.cat((torch.sin(x),(torch.cos(3*x)+torch.sin(2*x))/2,(torch.sin(3*x)+torch.sin(5*x))/2),dim = 1)
    
    
    
    
    num_nodes = 100
    net = nn.Sequential(
        nn.Linear(2, num_nodes),
        nn.Tanh(),
        nn.Linear(num_nodes,num_nodes),
        nn.Tanh(),
        nn.Linear(num_nodes,num_nodes),
        nn.Tanh(),
        nn.Linear(num_nodes,3))
    
    
    
    loss = SamplesLoss(loss="energy")
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    steps = int(5e4)
    batch_size = 500
    num_densities = 5
    min_val = -math.pi
    max_val = math.pi
    
    locs = np.random.uniform(-math.pi,math.pi,num_densities)
    # locs = np.linspace(-math.pi, math.pi, num_densities+1)[:-1]
    sd = np.random.uniform(1,3,num_densities)
    
    xs = []
    ys = []
    VMs = []
    for j in range(num_densities):
        vm = VonMises(
        loc=torch.tensor(locs[j], dtype=torch.float),
        concentration=sd[j])
        VMs.append(vm)
    
        
        
    for i in range(steps):
        
        optimizer.zero_grad()
        #### build training data 
        
        X = torch.zeros((num_densities,batch_size,d),dtype = torch.float)
        Y = torch.zeros((num_densities,batch_size,d),dtype = torch.float)
    
        for j in range(num_densities):
            x = VMs[j].sample((batch_size,1))
            y = ft(x)
            x_cat = torch.cat((torch.sin(x),torch.cos(x)),dim = 1)/torch.pi
            X[j] = net(x_cat)
            Y[j] = y
    
        L = loss(X,Y).mean()
       
        L.backward()
        
        optimizer.step()
            
    
        
        if i%100==0:
            print('Iteration: ',i, ' Loss: ', L.detach().numpy())
        if i%1000 == 0:
            
            test_size = int(1e4)
            
            cols = ['b','r','g','c','salmon','gold']
            fig,ax = plt.subplots(2,5,figsize = (20,7),dpi =300)
            for j in range(5):
                x = VMs[j].sample((test_size,1))
                y = ft(x)
                x_cat = torch.cat((torch.sin(x),torch.cos(x)),dim = 1)/torch.pi
    
                out = net(x_cat)
    
                kernel1 = stats.gaussian_kde(x.detach().numpy().T)
                kernel2 = stats.gaussian_kde(y[:,2].detach().numpy().T)
                kernel3 = stats.gaussian_kde(out[:,2].detach().numpy().T)       
                z = np.linspace(-math.pi,math.pi,1000)
                k1 = kernel1(z)
                k2 = kernel2(z)
                k3= kernel3(z)
                # ax[0,j].plot(z,k1,color = 'k',linewidth = 2,alpha = 0.25)
                # ax[1,j].plot(z,k2,color = 'k',linewidth = 2,alpha = 0.25)
                # ax[1,j].plot(z,k3,color = 'b',linewidth = 2,alpha = 0.5)
                ax[0,j].fill_between(z,0,k1,color= 'k',alpha = 0.2)
                ax[1,j].fill_between(z,0,k2,color = 'k',alpha = 0.2,label = 'Ground Truth')
                ax[1,j].fill_between(z,0,k3,facecolor='none', hatch = '//',edgecolor = 'b',label = 'Reconstructed')
                
            ax[1,4].legend(loc='upper right',
                            bbox_to_anchor=(1, 2.95),   # adjust as needed
                            fontsize=15,ncol=2 
                            )
            for ii in range(5):
                for jj in range(2):
                    ax[jj,ii].set_xlabel(r'$x$',fontsize = 15)
                    ax[jj,ii].set_ylabel(r'$y$',fontsize = 15)
                    ax[0,ii].set_xlim(-math.pi,math.pi)
                    ax[1,ii].set_xlim(-1,1)
    
            ax[0,0].set_title(r'$\rho_1$',fontsize = 25)
            ax[0,1].set_title(r'$\rho_2$',fontsize = 25)
            ax[0,2].set_title(r'$\rho_3$',fontsize = 25)
            ax[0,3].set_title(r'$\rho_4$',fontsize = 25)
            ax[0,4].set_title(r'$\rho_5$',fontsize = 25)
            
            ax[1,0].set_title(r'$f_3\#\rho_1$',fontsize = 25)
            ax[1,1].set_title(r'$f_3\#\rho_2$',fontsize = 25)
            ax[1,2].set_title(r'$f_3\#\rho_3$',fontsize = 25)
            ax[1,3].set_title(r'$f_3\#\rho_4$',fontsize = 25)
    
            ax[1,4].set_title(r'$f_3\#\rho_5$',fontsize = 25)
    
            plt.subplots_adjust(wspace = 0.4,hspace = 0.5)
            plt.show()
    
            x = torch.tensor(np.linspace(-math.pi,math.pi,1000),dtype = torch.float).unsqueeze(1)
            y = ft(x).detach().numpy()
            x_cat = torch.cat((torch.sin(x),torch.cos(x)),dim = 1)/torch.pi
    
            out = net(x_cat).detach().numpy()
            x = x.detach().numpy()
            fig,ax = plt.subplots(1,3,figsize = (12,3),dpi = 300)
            lw = 2
            ax[0].plot(x,y[:,0],color = 'k',linewidth = 2*lw,alpha = 0.25,label = 'Ground Truth')
            ax[0].plot(x,out[:,0],color = 'b',linewidth = lw,linestyle = '--',label = 'Recovered')
            
            ax[1].plot(x,y[:,1],color = 'k',linewidth = 2*lw,alpha = 0.25,label = 'Ground Truth')
            ax[1].plot(x,out[:,1],color = 'b',linewidth = lw,linestyle = '--',label = 'Recovered')
            
            ax[2].plot(x,y[:,2],color = 'k',linewidth = 2*lw,alpha = 0.25,label = 'Ground Truth')
            ax[2].plot(x,out[:,2],color = 'b',linewidth = lw,linestyle = '--',label = 'Recovered')
            ax[0].set_xlabel(r'$x$',fontsize = 15)
            ax[1].set_xlabel(r'$x$',fontsize = 15)
            ax[2].set_xlabel(r'$x$',fontsize = 15)
            ax[0].set_ylabel(r'$f_1(x)$',fontsize = 15)
            ax[1].set_ylabel(r'$f_2(x)$',fontsize = 15)
            ax[2].set_ylabel(r'$f_3(x)$',fontsize = 15)
            ax[0].legend(loc = 'upper left')
            plt.subplots_adjust(wspace  = 0.4)
            plt.show()
                
        
        
            
    test_size = int(1e4)
    
    cols = ['b','r','g','c','salmon','gold']
    fig,ax = plt.subplots(2,5,figsize = (20,7),dpi =300)
    for j in range(5):
        x = VMs[j].sample((test_size,1))
        y = ft(x)
        x_cat = torch.cat((torch.sin(x),torch.cos(x)),dim = 1)/torch.pi
    
        out = net(x_cat)
    
        kernel1 = stats.gaussian_kde(x.detach().numpy().T)
        kernel2 = stats.gaussian_kde(y[:,2].detach().numpy().T)
        kernel3 = stats.gaussian_kde(out[:,2].detach().numpy().T)       
        z = np.linspace(-math.pi,math.pi,1000)
        k1 = kernel1(z)
        k2 = kernel2(z)
        k3= kernel3(z)
        # ax[0,j].plot(z,k1,color = 'k',linewidth = 2,alpha = 0.25)
        # ax[1,j].plot(z,k2,color = 'k',linewidth = 2,alpha = 0.25)
        # ax[1,j].plot(z,k3,color = 'b',linewidth = 2,alpha = 0.5)
        ax[0,j].fill_between(z,0,k1,color= 'k',alpha = 0.2)
        ax[1,j].fill_between(z,0,k2,color = 'k',alpha = 0.2,label = 'Ground Truth')
        ax[1,j].fill_between(z,0,k3,facecolor='none', hatch = '//',edgecolor = 'b',label = 'Reconstructed')
        
    ax[1,4].legend(loc='upper right',
                    bbox_to_anchor=(1, 2.95),   # adjust as needed
                    fontsize=15,ncol=2 
                    )
    for ii in range(5):
        for jj in range(2):
            ax[jj,ii].set_xlabel(r'$x$',fontsize = 15)
            ax[jj,ii].set_ylabel(r'$y$',fontsize = 15)
            ax[0,ii].set_xlim(-math.pi,math.pi)
            ax[1,ii].set_xlim(-1,1)
    
    ax[0,0].set_title(r'$\rho_1$',fontsize = 25)
    ax[0,1].set_title(r'$\rho_2$',fontsize = 25)
    ax[0,2].set_title(r'$\rho_3$',fontsize = 25)
    ax[0,3].set_title(r'$\rho_4$',fontsize = 25)
    ax[0,4].set_title(r'$\rho_5$',fontsize = 25)
    
    ax[1,0].set_title(r'$f_3\#\rho_1$',fontsize = 25)
    ax[1,1].set_title(r'$f_3\#\rho_2$',fontsize = 25)
    ax[1,2].set_title(r'$f_3\#\rho_3$',fontsize = 25)
    ax[1,3].set_title(r'$f_3\#\rho_4$',fontsize = 25)
    
    ax[1,4].set_title(r'$f_3\#\rho_5$',fontsize = 25)
    
    plt.subplots_adjust(wspace = 0.4,hspace = 0.5)
    plt.show()
    
    x = torch.tensor(np.linspace(-math.pi,math.pi,1000),dtype = torch.float).unsqueeze(1)
    y = ft(x).detach().numpy()
    x_cat = torch.cat((torch.sin(x),torch.cos(x)),dim = 1)/torch.pi
    
    out = net(x_cat).detach().numpy()
    x = x.detach().numpy()
    fig,ax = plt.subplots(1,3,figsize = (12,3),dpi = 300)
    lw = 2
    ax[0].plot(x,y[:,0],color = 'k',linewidth = 2*lw,alpha = 0.25,label = 'Ground Truth')
    ax[0].plot(x,out[:,0],color = 'b',linewidth = lw,linestyle = '--',label = 'Recovered')
    
    ax[1].plot(x,y[:,1],color = 'k',linewidth = 2*lw,alpha = 0.25,label = 'Ground Truth')
    ax[1].plot(x,out[:,1],color = 'b',linewidth = lw,linestyle = '--',label = 'Recovered')
    
    ax[2].plot(x,y[:,2],color = 'k',linewidth = 2*lw,alpha = 0.25,label = 'Ground Truth')
    ax[2].plot(x,out[:,2],color = 'b',linewidth = lw,linestyle = '--',label = 'Recovered')
    ax[0].set_xlabel(r'$x$',fontsize = 15)
    ax[1].set_xlabel(r'$x$',fontsize = 15)
    ax[2].set_xlabel(r'$x$',fontsize = 15)
    ax[0].set_ylabel(r'$f_1(x)$',fontsize = 15)
    ax[1].set_ylabel(r'$f_2(x)$',fontsize = 15)
    ax[2].set_ylabel(r'$f_3(x)$',fontsize = 15)
    ax[0].legend(loc = 'upper left')
    plt.subplots_adjust(wspace  = 0.4)
    plt.show()
        



    with open("1D_test_{}.p".format(iii), "wb") as f:
        pickle.dump([z,k1,k2,k3,x,y,out], f)
      








        
        
        
        
