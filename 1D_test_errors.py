import numpy as np
import matplotlib.pyplot as plt
import pickle


errors = []

for i in range(10):

    with open("1D_test_{}.p".format(i), "rb") as f:
        data = pickle.load(f)
        z,k1,k2,k3,x,y,out = data
        
        
        errors.append(np.mean(np.sum((y-out)**2,axis = 1))/np.mean(np.sum(out**2,axis = 1)))
        plt.plot(out[:,1])
        plt.plot(y[:,1])

        plt.show()
        
print('mean and sd',np.mean(errors),np.std(errors))
        
print('min and max', np.min(errors),np.max(errors))  
print('median', np.median(errors))  

plt.plot()
