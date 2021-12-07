from collections import Counter
import numpy as np
from data_utils import *
from imblearn.over_sampling import BorderlineSMOTE

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation
from tensorflow.python.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, SGD

x_train,y_train=get_data(train=True,corrupt=True)
x_test,y_test=get_data(train=False)
sm=BorderlineSMOTE(random_state=42,kind="borderline-1")
x_train,y_train=sm.fit_resample(x_train,y_train)
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

history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=0)
print("neural networks:")
score=model.evaluate(x_test,y_test,batch_size=128)
print("acc:"+str(score[1]))