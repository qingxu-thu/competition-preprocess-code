import pandas as pd
import os
import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import f1_score
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import Adam
from collections import defaultdict


def txtread(path):
    df_0702 = pd.read_csv(f'{path}', sep=';', names=['linkid_' 'label_' 'current_slice_id_' 'future_slice_id', 'recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4'])
    df_test = pd.DataFrame(columns=['linkid', 'label', 'current_slice_id', 'future_slice_id'])
    df_test[['linkid', 'label', 'current_slice_id', 'future_slice_id']] = df_0702.iloc[:, 0].str.split(' ', expand=True)
    df_test = df_test.astype('int')
    df_test[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']] = df_0702[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']]
    return df_test


def get_feature(df):
    ##将recent_feature, history_feature1-4,合并并且储存成lstm所能使用格式。
    ##输入df
    ##输出label，以及(-1, 5, 20)的numpy数组。

    def get_split(df_t):
        arr_list = []
        print('start a new split, please wait patiently, it takes about two minues to get done')
        for i in range(5):
            f = df_t[i].str.split(':', expand=True)[1].str.split(',', expand=True).astype(float).values.reshape((-1, 1, 4))
            arr_list.append(f)
        return np.concatenate(arr_list, axis=1)
    recent_feature = df.recent_feature.str.split(' ', expand=True)
    recent_feature = get_split(recent_feature)

    history_feature1 = df.history_feature1.str.split(' ', expand=True)
    history_feature1 = get_split(history_feature1)

    history_feature2 = df.history_feature2.str.split(' ', expand=True)
    history_feature2 = get_split(history_feature2)

    history_feature3 = df.history_feature3.str.split(' ', expand=True)
    history_feature3 = get_split(history_feature3)

    history_feature4 = df.history_feature4.str.split(' ', expand=True)
    history_feature4 = get_split(history_feature4)
    
    combined_feature = np.concatenate([recent_feature, history_feature1, history_feature2, history_feature3, history_feature4], axis=2)
    combined_feature = combined_feature.reshape([-1,100])
    return combined_feature, df.label


def read_link_feature(link_path,connect_path):
    link_attr = {}
    f = open(link_path,'r')
    for line in f.readlines():
        #print(line)
        attr = line.split("\t")
        attr = [float(a) for a in attr]
        if len(attr)<9:
            attr.append(0)
        link_attr[int(attr[0])]=attr[1:]
    f = open(connect_path,'r')
    link_connection = {}
    for line in f.readlines():
        low = line.split("\t")
        key = int(low[0])
        low = low[1]
        low = low.split(",")
        low = [int(a) for a in low]
        link_connection[key] = low[:]
    
    inv_link = defaultdict(list)
    for key, value in link_connection.items():
        for id_v in value:
            inv_link[id_v].append(key)
    
    return link_attr, link_connection, inv_link


def get_attr_mean(link_id, link_attr, link_connection, inv_link):
    if link_id in link_connection:
        #print(link_connection[link_id])
        for i, low in enumerate(link_connection[link_id]):
            #print(low,link_attr[low])
            attr_low = link_attr[low]
            if i == 0:
                attr = np.array(attr_low)
            else:
                attr += np.array(attr_low)
        attr = attr/(i+1)
    else:
        attr = np.zeros((8))
    if link_id in inv_link:
        for i, low in enumerate(inv_link[link_id]):
            attr_low = link_attr[low]
            if i == 0:
                attr2 = np.array(attr_low)
            else:
                attr2 += np.array(attr_low)
        attr2 = attr/(i+1)
    else:
        attr2 = np.zeros((8))
    return np.concatenate((attr,attr2),axis=0)


def save_link_feature(link_attr, link_connection, inv_link):
    i = 0
    feature = np.zeros((len(link_attr),16))
    for key,value in enumerate(link_attr):
        #print(key)
        link_id = int(key)
        #feature = get_attr_mean(link_id, link_attr, link_connection, inv_link)
        #if i == 0:
        #    feature = get_attr_mean(link_id, link_attr, link_connection, inv_link)        
        #else: 
        #    feature = np.concatenate((feature,get_attr_mean(link_id, link_attr, link_connection, inv_link)),axis=1)
        
        feature[i] = get_attr_mean(link_id, link_attr, link_connection, inv_link)
        i+=1
    np.save('./link_feature.npy',feature)
    return feature

def derive_link_feature(path):
    return np.load(path)

def get_link_feature(df, feature,combined_feature):
    link_id = df['linkid']
    print(combined_feature.shape,feature[link_id,:].shape)
    return np.concatenate((combined_feature,feature[link_id,:]),axis=1)

'''
def get_link_feature(df, link_attr, link_connection, inv_link, combined_feature):
    i = 0
    for link_id in df['linkid']:
        link_id = int(link_id)
        if i == 0:
            feature = get_attr_mean(link_id, link_attr, link_connection, inv_link)
        else: 
            feature = np.concatenate((feature,get_attr_mean(link_id, link_attr, link_connection, inv_link)),axis=1)
        i+=1
        print(i)
    print(combined_feature.shape,feature.shape)
    total_feature = np.concatenate((combined_feature,feature),axis=0)
    return total_feature
'''
    
def build_model(input_size,layers):
    model = Sequential()
    a = int(math.log(input_size))
    for i in range(layers):
        if i==0:
            model.add(Dense(math.pow(2,a),input_shape=(input_size,),activation='relu'))
            model.add(Dropout(0.2))
        else:
            model.add(Dense(math.pow(2,a-i),activation='relu'))
            model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.001))
    return model


class Metrics(keras.callbacks.Callback):
    def __init__(self, val_data, batch_size = 64):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_f1s = []
 
    def on_epoch_end(self, epoch, logs={}):
        print(self.validation_data)
        val_predict = (np.asarray(self.model.predict(self.validation_data[0])))
        #print(val_predict)
        val_predict = np.argmax(val_predict,axis=1)
        print(val_predict.shape)
        val_targ = self.validation_data[1]
        #print(val_targ[val_targ==1],val_targ[val_targ==2])
        _val_f1 = f1_score(val_targ, val_predict, average=None)
        #print(_val_f1)
        _val_f1 = _val_f1[0]*0.2+_val_f1[1]*0.2+_val_f1[2]*0.6
        self.val_f1s.append(_val_f1)
        print(" — val_f1: %f " % (_val_f1))
        return
 

def model_train(model,x_train,y_train,batch_size,epochs,x_test,y_test, metrics):
    #history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_test, y_test),callbacks=[metrics])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=0.1,callbacks=[metrics])
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss",score[0])


def multiple_file_read(filePath,num,link_attr, link_connection, inv_link):
    path_list = os.listdir(filePath)
    for i in range(num):
        print(i)
        path = path_list[i]
        df = txtread(filePath+path)
        if i == 0:
            combined_feature, label = get_feature(df)
            label = np.array(label)
            #feature = save_link_feature(link_attr, link_connection, inv_link)
            feature = derive_link_feature("./link_feature.npy")
            print(feature.shape)
            x_train = get_link_feature(df, feature, combined_feature)
        else:
            combined_feature, a = get_feature(df)
            label = np.concatenate((label,a),axis=0)
            total_feature_a = get_link_feature(df, feature, combined_feature)
            x_train = np.concatenate((x_train,total_feature_a),axis=0)
    return x_train,label


link_path = '../20201012150828attr.txt'
connect_path = '../20201012151101topo.txt'
link_attr, link_connection, inv_link = read_link_feature(link_path,connect_path)
num = 5
filePath = '../traffic-fix (1)/traffic/'
x_train,label = multiple_file_read(filePath,num,link_attr, link_connection, inv_link)
label[label==1]=0
label[label==2]=1
label[label==3]=2
label[label==4]=2
index = np.array([i for i in range(x_train.shape[0])])
np.random.shuffle(index)
train_num = int(x_train.shape[0]*9/10)
model = build_model(x_train.shape[1],4)
batch_size = 64
epochs = 20
metrics = Metrics((x_train[index[train_num:train_num+int(train_num/9)]],label[index[train_num:train_num+int(train_num/9)]]))
model_train(model,x_train[index[:train_num]],label[index[:train_num]],batch_size,epochs,x_train[index[train_num:]],label[index[train_num:]], metrics)

