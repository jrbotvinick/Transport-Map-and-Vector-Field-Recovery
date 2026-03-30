import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle 
import math

def eval_gauss(x,center,sigma):
    return (1/(2*math.pi*sigma**2) ) * np.exp(- ((x[:,0]-center[0])**2+(x[:,1]-center[1])**2)/(2*sigma**2))

fig,ax = plt.subplots(3,5,figsize = (20,10),dpi = 300)

ms = [1,2,3,4,5,6,7,8]
N = 100
x,y = np.linspace(-1,1,N),np.linspace(-1,1,N)

counter = 0

errs = []

for m in ms:
    with open("divergence_experiment_{}.p".format(m), "rb") as f:
        data = pickle.load(f)
        v,v_net,XX,centers,sigmas = data
        
        
    diff = np.sum((v-v_net)**2,axis = 1)
    errs.append(np.mean(diff)/np.mean(np.sum(v**2,axis = 1)))
    print(np.mean(diff))

    gs = []
    for j in range(m):
        gs.append(eval_gauss(XX,centers[j],sigmas[j]))
        
    gs = np.array(gs)
    g = np.max(gs,axis = 0)
    
    mag = np.sqrt(np.sum(v_net**2,axis = 1))
    if counter < 5:
        c0 = ax[0,counter].contourf(x,y,g.reshape(N,N),levels = 30,vmin = 0.05,vmax = .159)
        ax[0,counter].scatter(centers[:m,0],centers[:m,1],color = 'k',marker = 'x')
        
        ax[1,counter].streamplot(XX[:,0].reshape(N,N),XX[:,1].reshape(N,N),v_net[:,0].reshape(N,N),v_net[:,1].reshape(N,N),density = 2,linewidth = 0.5,arrowsize = 0.5,color = 'k')
        c1 = ax[1,counter].contourf(x,y,mag.reshape(N,N),alpha = 0.5,cmap = 'RdBu_r',vmin = 0,vmax = 1.5)

        if counter == 1:
            c2 = ax[2,counter].contourf(x,y,diff.reshape(N,N),levels = 100,vmin = 0,vmax = 1,cmap = 'jet')
        else:
            ax[2,counter].contourf(x,y,diff.reshape(N,N),levels = 100,vmin = 0,vmax = 1,cmap = 'jet')

        counter += 1
 
    
plt.subplots_adjust(wspace = 0.4,hspace = 0.3)

for j in range(3):
    for k in range(5):
        ax[j,k].set_xlabel(r'$x$',fontsize= 12)
        ax[j,k].set_ylabel(r'$y$',fontsize= 12)

for j in range(5):
    ax[0,j].set_title(r'$m = {}$'.format(j+1),fontsize = 20)
    
    
fig.text(0.05,0.78,r'Densities $\{\rho_j\}$',rotation=90,
         va='center',fontsize=20)

fig.text(0.05,0.50,r'Learned $v_{\theta}$',
         rotation=90,va='center',fontsize=20)

fig.text(0.05,0.22,r'Squared Error$',
         rotation=90,va='center',fontsize=20)

cbar0 = fig.colorbar(c0, ax=ax[0,:], fraction=0.02, pad=0.02)
cbar1 = fig.colorbar(c1, ax=ax[1,:], fraction=0.02, pad=0.02)
cbar2 = fig.colorbar(c2, ax=ax[2,:], fraction=0.02, pad=0.02)
cbar0.set_label(r'$\max \{\rho_j(x)\}_{j=1}^m$', fontsize=14)
cbar1.set_label(r'Magnitude', fontsize=14)
cbar2.set_label(r'Squared error', fontsize=14)


plt.show()

mag =  np.sum(v**2,axis = 1)
mag2 = np.sum(v_net**2,axis = 1)
fig,ax = plt.subplots(1,3,figsize = (10,3),dpi = 300)
ax[0].streamplot(XX[:,0].reshape(N,N),XX[:,1].reshape(N,N),v[:,0].reshape(N,N),v[:,1].reshape(N,N),density = 2,linewidth = 0.5, arrowsize = 0.5,color = 'k')
ax[0].contourf(x,y,mag.reshape(N,N),alpha = 0.5,cmap = 'RdBu_r',vmin = 0,vmax = 1.5)

ax[1].streamplot(XX[:,0].reshape(N,N),XX[:,1].reshape(N,N),v_net[:,0].reshape(N,N),v_net[:,1].reshape(N,N),density = 2,linewidth = 0.5, arrowsize = 0.5,color = 'k')
ax[1].contourf(x,y,mag2.reshape(N,N),alpha = 0.5,cmap = 'RdBu_r',vmin = 0,vmax = 1.5)

for i in range(2):
    ax[i].set_xlim(-1,1)
    ax[i].set_ylim(-1,1)
    ax[i].set_xlabel(r'$x$',fontsize = 12)
    ax[i].set_ylabel(r'$y$',fontsize = 12)

    
    
ax[2].plot(ms,errs,'--o',color = 'k')
ax[2].set_yscale('log')

ax[2].set_xlabel(r'Number of Densities ($m$)',fontsize = 12)
ax[2].set_ylabel(r'Relative error',fontsize = 12)
ax[0].set_title(r'$v$(x,y)',fontsize = 12)
ax[1].set_title(r'$v_{\theta}(x,y)$',fontsize = 12)
ax[2].set_xticks(ms)
plt.subplots_adjust(wspace = 0.4)
plt.show()



