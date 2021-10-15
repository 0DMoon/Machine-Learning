from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import multiclass


def get_data(filename):
    x=[]
    y=[]
    with open(filename,'r') as fp:
        lines=fp.readlines()
    for i in range(len(lines)):
        line=lines[i].strip().split(' ')
        line=[x for x in line if x!=' ' and x!='']
        x.append([float(line[i]) for i in range(1,len(line)-1)])
        y.append(line[len(line)-1])
    return x,y

data,label=get_data('yeast.data')
# print(data,label)
# print(x_train)
log_reg=LogisticRegression()
log_reg1=LogisticRegression(multi_class='multinomial',solver='newton-cg')
ovr=OneVsRestClassifier(log_reg)
ovo=OneVsOneClassifier(log_reg1)

rate=[]
res_ovr=[]
res_ovo=[]
for i in range(1,10):
    rate.append(0.1*i)
    x_train,x_test,y_train,y_test=train_test_split(data,label,random_state=1,test_size=1-0.1*i)
    ovr.fit(x_train,y_train)
    res_ovr.append(ovr.score(x_test,y_test))
    ovo.fit(x_train,y_train)
    res_ovo.append(ovo.score(x_test,y_test))
    print("rate:",0.1*i)
    print("ovr:",ovr.score(x_test,y_test))
    print("ovo:",ovo.score(x_test,y_test))

plt.plot(rate,res_ovr,color='blue',label='ovr')
plt.plot(rate,res_ovo,color='red',label='ovo')
plt.legend(loc='upper left')
plt.xlabel('rate')
plt.axis([0,1,0,1])
plt.show()