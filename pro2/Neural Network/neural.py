import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation
from tensorflow.python.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, SGD

def get_data(filename):
    x=[]
    y=[]
    with open(filename,'r') as fp:
        lines=fp.readlines()
    for i in range(len(lines)):
        line=lines[i].strip().replace(' ','').split(',')
        x.append([float(line[i]) for i in range(16)])
        y.append(float(line[16]))
    return np.array(x),np.array(y)

x_train,y_train=get_data('pendigits.tra')
x_test,y_test=get_data('pendigits.tes')
# print(x_train)

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

history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=1)
score=model.evaluate(x_test,y_test,batch_size=128)
# print()
# print(score[0])
# print(1)
