import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from mpl_toolkits.mplot3d import Axes3D
from geomloss import SamplesLoss
loss = SamplesLoss(loss="energy")


errors_v = []
errors_f = []
N_samples = int(1e5)
init = [-10,10,40]
X0 = init+np.random.normal(0,7,(N_samples,3))


for ii in range(10):
    
    with open("lorenz_experiment_{}.p".format(ii), "rb") as f:
        data = pickle.load(f)
        
    dt,N_evolve,Xs,M = data[0], data[1], data[2], data[3]
    
    
    fig,ax = plt.subplots(1,8,figsize = (25,3),dpi = 300)
    for i in range(8):
        ax[i].set_title('t = {:.1f}'.format(i*N_evolve*dt),fontsize = 20)
        im =  ax[i].hist2d(Xs[i][:N_samples,0]*M,Xs[i][:N_samples,2]*M,range = [[-25,25],[0,50]],bins = 100,density = True,vmin = 0,vmax = 0.005)
        ax[i].set_xlabel(r'$x$',fontsize = 20)
    ax[0].set_ylabel(r'$z$',fontsize = 20)
    
    
    cax = fig.add_axes([0.91, 0.15, 0.005, 0.7])
    fig.colorbar(im[3], cax=cax, label="Density")
    
    fig.text(0.085, 0.5, "Training Data", va='center', ha='left', fontsize=22, rotation=90)
    
    plt.subplots_adjust(wspace = 0.2,hspace = 0.5)
    plt.show()
    
    def lorenz_vec(xyz, *, s=10, r=28, b=2.667):
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot]).T
    
    def lorenz(xyz, *, s=10, r=28, b=2.667):
        x, y, z = xyz
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])
    
    
    def evolve(rho):
        X = rho.copy()
        for i in range(N_evolve):
            X = lorenz_vec(X)*dt+X
        return X
    
    def evolve_NN(X,f):
        with torch.no_grad():
            Y = X.clone()
            for i in range(N_evolve):    
                Y = f(Y)*dt+Y
            return Y
    
    
    net = nn.Sequential(
        nn.Linear(3, 100),
        nn.Tanh(),
        nn.Linear(100,100),
        nn.Tanh(),
        nn.Linear(100,100),
        nn.Tanh(),
        nn.Linear(100,3))
    
    net.load_state_dict(torch.load("lorenz_pushforward_{}.pth".format(ii)))
    net.eval()
    
    
    
    
    X = X0
    # X = Xs[-1][:N_samples]*M
    N_pred = 7
    Y = torch.tensor(X,dtype = torch.float)/M
    Zs = []
    Ws = []
    W = X
    Zs.append(X)
    Ws.append(X)
    for k in range(N_pred):  
        W = evolve(W)
        Ws.append(W)
        Y = evolve_NN(Y,net)
        Z = Y.detach().numpy()*M
        Zs.append(Z)
      
    
    fig,ax = plt.subplots(2,N_pred+1,figsize = (25,6),dpi = 300)
    for i in range(N_pred+1):
        ax[0,i].set_title('t = {:.1f}'.format(i*N_evolve*dt),fontsize = 20)
    
        im = ax[0,i].hist2d(Ws[i][:,0],Ws[i][:,2],range = [[-25,25],[0,50]],bins = 100,density = True,vmin = 0,vmax = 0.005)
        im2 = ax[1,i].hist2d(Zs[i][:,0],Zs[i][:,2],range = [[-25,25],[0,50]],bins = 100,density = True,vmin = 0,vmax = 0.005)
        ax[1,i].set_xlabel(r'$x$',fontsize = 20)
    ax[0,0].set_ylabel(r'$z$',fontsize = 20)
    ax[1,0].set_ylabel(r'$z$',fontsize = 20)
    
    
    cax = fig.add_axes([0.91, 0.15, 0.005, 0.7])
    fig.colorbar(im[3], cax=cax, label="Density")
    fig.colorbar(im2[3], cax=cax, label="Density")
    
    fig.text(0.085, 0.7, "Ground truth", va='center', ha='left', fontsize=22, rotation=90)
    fig.text(0.085, 0.3, "Prediction",   va='center', ha='left', fontsize=22, rotation=90)
    
    plt.subplots_adjust(wspace = 0.2,hspace = 0.2)
    plt.show()
    
    
    
    
    y = np.array([  8.05441154,   7.90649224,  26.82471414])
    ys = []
    for k in range(int(1e5)):
        ys.append(y)
        y = y + dt*lorenz(y)
    ys = np.array(ys)
    
    z = Xs[:-1].reshape((7*int(1e5),3))*M
    z = np.random.permutation(z)
    ys = np.random.permutation(ys)

    nplot = int(1e3)
    nplot2 = int(1e5)
    V = lorenz_vec(ys)
    V2 = lorenz_vec(z)

    mags = np.sqrt(np.sum(V**2,axis = 1))
    norm = plt.Normalize(vmin=mags.min(), vmax=mags.max()/2)
    colors = plt.cm.jet(norm(mags))
    
    
    V_NN = net(torch.tensor(ys,dtype = torch.float)/M)*M
    V_NN = V_NN.detach().numpy()
    
    V_NN2 = net(torch.tensor(z,dtype = torch.float)/M)*M
    V_NN2 = V_NN2.detach().numpy()
    
    
    ## compute flow error
    es = []
    
    pred = evolve_NN(torch.tensor(Xs[:-1].reshape((7*int(1e5),3)),dtype = torch.float),net).detach().numpy()*M
    truth = evolve(M*Xs[:-1].reshape((7*int(1e5),3)))
    
    error = np.mean((np.sum((pred-truth)**2,axis = 1)))/np.mean((np.sum(truth**2,axis = 1)))
    # for k in range(7):
    #     pred = evolve_NN(torch.tensor(Xs[k],dtype = torch.float),net).detach().numpy()*M
    #     truth = evolve(Xs[k]*M)
    #     error = np.mean((np.sum((pred-truth)**2,axis = 1)))/np.mean((np.sum(truth**2,axis = 1)))
    #     es.append(error)

    
    print('Pred_err: ', error)
    errors_f.append(error)
    
    
    
    
    
    
    v_err = np.sum((V_NN-V)**2,axis = 1)/np.sum(V**2,axis = 1)
    norm2 = plt.Normalize(vmin=0,vmax = 1)
    colors_err = plt.cm.jet(norm2(v_err))

    
    v_err2 = np.mean((np.sum((V_NN2-V2)**2,axis = 1)))/np.mean(((np.sum((V2)**2,axis = 1))))
    
    errors_v.append(v_err2)
    print('vector field error: ', v_err2)
    
    
    mags_NN = np.sqrt(np.sum(V**2,axis = 1))
    mags_norm_NN = (mags_NN - mags.min()) / (mags.max()-mags.min())
    colors2 = plt.cm.jet(norm(mags_NN))
    
    fig = plt.figure(figsize=(10,8), dpi=300, constrained_layout=True)

    gs = fig.add_gridspec(2, 2, width_ratios=[1,1], height_ratios=[1,1], wspace=0.0, hspace= 0)
    
    # 3D subplots
    ax1 = fig.add_subplot(gs[0,0], projection='3d')  # Density samples
    ax2 = fig.add_subplot(gs[0,1], projection='3d')  # Error magnitude
    ax3 = fig.add_subplot(gs[1,0], projection='3d')  # True velocity
    ax4 = fig.add_subplot(gs[1,1], projection='3d')  # Learned velocity
    
    # Manual short/thin colorbars
    cax_err = fig.add_axes([1.0, 0.63, 0.015, 0.25])  # relative error
    cax_mag = fig.add_axes([1.0, 0.14, 0.015, 0.25])

    ax1.set_title('Data Support',fontsize= 15,y = 0.92)
    ax1.set_xlabel(r'$x$',fontsize= 15)
    ax1.set_ylabel(r'$y$',fontsize= 15)
    ax1.set_zlabel(r'$z$',fontsize= 15)
    
    ax1.scatter(z[:,0][:nplot2],z[:,1][:nplot2],z[:,2][:nplot2],color = 'k',s = 0.005,alpha =0.5)
    ax1.view_init(15,110)
    ax1.set_xlim(-25,25)
    ax1.set_ylim(-25,25)
    ax1.set_zlim(0,50)
    ax1.xaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax1.yaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax1.zaxis.set_tick_params(labelsize=8,pad = 0.5)
    ax1.grid(False)

    ax2.set_title('Reconstruction Error',fontsize= 15,y = 0.92)
    ax2.set_xlabel(r'$x$',fontsize= 15)
    ax2.set_ylabel(r'$y$',fontsize= 15)
    ax2.set_zlabel(r'$z$',fontsize= 15)
    
    sc= ax2.scatter(ys[:,0],ys[:,1],ys[:,2],s = 0.01,c = colors_err,cmap = 'jet')
    ax2.view_init(15,110)
    ax2.set_xlim(-25,25)
    ax2.set_ylim(-25,25)
    ax2.set_zlim(0,50)
    ax2.xaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax2.yaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax2.zaxis.set_tick_params(labelsize=8,pad = 0.5)
    ax2.grid(False)
    se = plt.cm.ScalarMappable(cmap='jet', norm=norm2)
    fig.colorbar(se, cax=cax_err, label="Relative Error")

    ax3.set_title('True Vector Field',fontsize= 15,y = 0.92)
    ax3.set_xlabel(r'$x$',fontsize= 15)
    ax3.set_ylabel(r'$y$',fontsize= 15)
    ax3.set_zlabel(r'$z$',fontsize= 15)
    
    ax3.quiver(ys[:,0][:nplot],ys[:,1][:nplot],ys[:,2][:nplot],V[:,0][:nplot],V[:,1][:nplot],V[:,2][:nplot],colors = colors[:nplot],length = 3,linewidth = 0.7,normalize = True)
    ax3.view_init(15,110)
    ax3.set_xlim(-25,25)
    ax3.set_ylim(-25,25)
    ax3.set_zlim(0,50)
    ax3.xaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax3.yaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax3.zaxis.set_tick_params(labelsize=8,pad = 0.5)
    ax3.grid(False)

    
    
    
    im = ax4.quiver(ys[:,0][:nplot],ys[:,1][:nplot],ys[:,2][:nplot],V_NN[:,0][:nplot],V_NN[:,1][:nplot],V_NN[:,2][:nplot],colors = colors2[:nplot],length = 3,linewidth = 0.7,normalize = True)
    ax4.set_title('Learned Vector Field',fontsize= 15,y = 0.92)
    ax4.set_xlabel(r'$x$',fontsize= 15)
    ax4.set_ylabel(r'$y$',fontsize= 15)
    ax4.set_zlabel(r'$z$',fontsize= 15)
    ax4.view_init(15,110)
    ax4.set_xlim(-25,25)
    ax4.set_ylim(-25,25)
    ax4.set_zlim(0,50)
    ax4.grid(False)

    ax4.xaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax4.yaxis.set_tick_params(labelsize=8, pad = 0.5)
    ax4.zaxis.set_tick_params(labelsize=8,pad = 0.5)
    
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    fig.colorbar(sm, cax=cax_mag, label="Velocity Magnitude")

    plt.show()


    
    

print(np.median(errors_v),np.min(errors_v),np.max(errors_v))

print(np.median(errors_f),np.min(errors_f),np.max(errors_f))


