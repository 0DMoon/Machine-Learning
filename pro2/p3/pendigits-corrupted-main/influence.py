import numpy as np
from data_utils import *
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation
from tensorflow.python.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, SGD
from sklearn import svm
from sklearn import tree

def main():
    noise=0
    x=[]
    y0,y1,y2=[],[],[]
    for i in range(11):
        x.append(noise)
        res=base(noise=noise)
        y0.append(res[0][1])
        y1.append(res[1][1])
        y2.append(res[2][1])
        noise+=0.1
    imb=1
    a=[]
    b0,b1,b2=[],[],[]
    for i in range(10):
        a.append(str(imb))
        res=base(imb=imb)
        b0.append(res[0][1])
        b1.append(res[1][1])
        b2.append(res[2][1])
        imb=imb+2
    
    for i in range(2,5):
        a.append(str(10**i))
        res=base(imb=10**i)
        b0.append(res[0][1])
        b1.append(res[1][1])
        b2.append(res[2][1])
    
    gs=gridspec.GridSpec(1,2)
    ax1=plt.subplot(gs[0,0])
    ax2=plt.subplot(gs[0,1])

    ax1.plot(x,y0,color='red',label="neural network")
    ax1.plot(x,y1,color='blue',label="svm")
    ax1.plot(x,y2,color='yellow',label="decision tree")
    ax1.set_title("Noise")
    ax1.set_xlabel("noise")
    ax1.set_ylabel("acc")
    ax1.legend()

    ax2.plot(a,b0,color='red',label="neural network")
    ax2.plot(a,b1,color='blue',label="svm")
    ax2.plot(a,b2,color='yellow',label="decision tree")
    ax2.set_title("Imbalance")
    ax2.set_xlabel("imb")
    ax2.set_ylabel("acc")
    ax2.legend()

    plt.show()

def base(noise=0.3,imb=10):
    x_train,y_train=get_data(train=True,corrupt=True,noise=noise,imb=imb)
    x_test,y_test=get_data(train=False)

    ###neural network###
    # sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model=Sequential()
    model.add(Dense(units=256,input_dim=16,activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(units=128,activation='relu'))
    # model.add(Dense(units=128,activation='relu'))
    # model.add(Dense(units=128,activation='relu'))
    # model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=16,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

    t0=time.time()
    history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=0)
    t1=time.time()
    score=model.evaluate(x_test,y_test,batch_size=128,verbose=0)
    cost0=t1-t0
    acc0=score[1]


    ###svm###
    clf=svm.SVC(gamma='scale',decision_function_shape='ovo')
    t0=time.time()
    clf.fit(x_train,y_train)
    t1=time.time()
    cost1=t1-t0

    y_prid=clf.predict(x_test)
    correct=0
    for i in range(len(y_prid)):
        if y_prid[i]==y_test[i]:
            correct+=1
    acc1=correct/len(y_test)


    ###decision tree###
    dt=tree.DecisionTreeClassifier()
    t0=time.time()
    dt.fit(x_train,y_train)
    t1=time.time()
    cost2=t1-t0

    y_prid=dt.predict(x_test)
    correct=0
    for i in range(len(y_prid)):
        if y_prid[i]==y_test[i]:
            correct+=1
    acc2=correct/len(y_test)

    return [[cost0,acc0],[cost1,acc1],[cost2,acc2]]
    

if __name__=='__main__':
    main()