import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calmatrix(threshold,y,yc):
    length=len(y)
    TP=FP=TN=FN=0
    for j in range(length):
        if (y[j]==1)and(yc[j]>=threshold):
            TP+=1
        elif (y[j]==1)and(yc[j]<threshold):
            FN+=1
        elif (y[j]==0)and(yc[j]>=threshold):
            FP+=1
        else:
            TN+=1
    return TP,FP,TN,FN

y=[1,0,1,1,1,0,0,1]
yc1=[0.62,0.39,0.18,0.72,0.45,0.01,0.32,0.93]
yc2=[0.34,0.12,0.82,0.89,0.17,0.75,0.36,0.97]

length = len(y)
TPR=np.zeros(length)
FPR=np.zeros(length)
syc1=yc1[:]
syc1.sort(reverse=True)
syc2=yc2[:]
syc2.sort(reverse=True)

for i in range(length):
    TP,FP,TN,FN=calmatrix(syc1[i],y,yc1)
    TPR[i]=TP/(TP+FN)
    FPR[i]=FP/(FP+TN)

auc=0
for i in range(length-1):
    auc+=(TPR[i]+TPR[i+1])*(FPR[i+1]-FPR[i])/2
print(auc)

plt.plot(FPR,TPR)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC of yc1")
plt.axis([0,1,0,1])
plt.show()

for i in range(length):
    TP,FP,TN,FN=calmatrix(syc2[i],y,yc2)
    TPR[i]=TP/(TP+FN)
    FPR[i]=FP/(FP+TN)

auc=0
for i in range(length-1):
    auc+=(TPR[i]+TPR[i+1])*(FPR[i+1]-FPR[i])/2
print(auc)

plt.plot(FPR,TPR)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC of yc2")
plt.axis([0,1,0,1])
plt.show()

TP,FP,TN,FN=calmatrix(0.4,y,yc1)
print(TP,FP,TN,FN)
TP,FP,TN,FN=calmatrix(0.9,y,yc2)
print(TP,FP,TN,FN)