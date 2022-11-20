from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


mat_dict = {}
mat_dict.update(loadmat('4k_CMAB_regret.mat'))


data=(mat_dict['Regret_record'])


n=data[0,0].shape[1]

x= list(range(data[0,0].shape[1]))

y=list(data[0,0][0,:])


data2=[]


def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]
 


for i in range(100):
    
    raw_data=list(data[i,0][0])

    data2.append(Cumulative(raw_data))
    



for i in range(20):
    
 
    
    x= list(range(len(data2[i])))
    y=data2[i]
    
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    
    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #        title='About as simple as it gets, folks')
    # ax.grid()
    
    # fig.savefig("test.png")
  
    
    plt.plot(x, y)




plt.show()