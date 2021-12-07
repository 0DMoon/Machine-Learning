import numpy as np
from data_utils import *
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation
from tensorflow.python.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, SGD
from sklearn import svm
from sklearn import tree

x_train,y_train=get_data(train=True,corrupt=True)
x_test,y_test=get_data(train=False)
# print(x_train.shape)
# print(y_train.shape)

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
print("neural networks:")
score=model.evaluate(x_test,y_test,batch_size=128)
print("acc:"+str(score[1]))
cost=t1-t0
print("cost:"+str(cost))


###svm###
clf=svm.SVC(gamma='scale',decision_function_shape='ovo')
t0=time.time()
clf.fit(x_train,y_train)
t1=time.time()
cost=t1-t0

y_prid=clf.predict(x_test)
correct=0
for i in range(len(y_prid)):
    if y_prid[i]==y_test[i]:
        correct+=1
acc=correct/len(y_test)
print("svm:")
print("acc:"+str(acc))
print("cost:"+str(cost))

###decision tree###
dt=tree.DecisionTreeClassifier()
t0=time.time()
dt.fit(x_train,y_train)
t1=time.time()
cost=t1-t0

y_prid=dt.predict(x_test)
correct=0
for i in range(len(y_prid)):
    if y_prid[i]==y_test[i]:
        correct+=1
acc=correct/len(y_test)
print("decision tree")
print("acc:"+str(acc))
print("cost:"+str(cost))