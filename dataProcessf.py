import pandas as pd
import os
import numpy as np
import joblib
import keras
from sklearn.metrics import f1_score
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.optimizers import Adam

def txtread(path):
    df_0702 = pd.read_csv(f'train/traffic/{path}', sep=';', names=['linkid_' 'label_' 'current_slice_id_' 'future_slice_id', 'recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4'])
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
    
    return combined_feature, df.label


def get_link_feature(link_path,connect_path):
    link_attr = {}
    f = os.open(link_path)
    for line in f.readlines():
        attr = line.spilt(" ")
        attr = [float(a) for a in attr]
        if len(attr)<9:
            attr.append(0)
        link_attr[attr[0]]=attr[1:-1]
    f = os.open(connect_path)
    link_connection = {}
    for line in f.readlines():
        low = line.spilt(" ")
        low = [int(a) for a in low]
        link_connection[low[0]] = low[1:-1]
    inv_link = {}
    for k, v in link_connection.iteritems():
        inv_link[v] = inv_map.get(v, [])
        inv_link[v].append(k)
    return link_attr, link_connection, inv_link


def get_attr_mean(link_id, link_attr, link_connection, inv_link):
    if link_id in link_connection:
        for i, low in enumerate(link_connection[link_id]):
            if i == 0:
                attr = np.array(low)
            else:
                attr += np.array(low)
        attr = attr/(i+1)
    else:
        attr = np.zeros((9))
    if link_id in inv_link:
        for i, low in enumerate(inv_link[link_id]):
            if i == 0:
                attr2 = np.array(low)
            else:
                attr2 += np.array(low)
        attr2 = attr/(i+1)
    else:
        attr2 = np.zeros((9))
    return np.concatenate((attr,attr2),axis=0)


def get_link_feature(df, ink_attr, link_connection, inv_link, combined_feature):
    i = 0
    for link_id in df['linkid']:
        link_id = int(link_id)
        if i == 0:
            feature = get_attr_mean(link_id, link_attr, link_connection, inv_link)
        else: 
            feature = np.concatenate((feature,get_attr_mean(link_id, link_attr, link_connection, inv_link)),axis = 1)
    total_feature = np.concatenate((combined_feature,feature),axis=0)
    return total_feature

    
def build_model(input_size,layers):
    model.Sequential()
    a = int(log(input_size))
    for i in range(layers):
        if i==0:
            model.add(Dense(math.pow(2,a),input_shape=(input_size,),activation='relu'))
            model.add(Dropout(0.2))
        else:
            model.add(Dense(math.pow(2,a-i),activation='relu'))
            model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=Adam())
    return model

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='None')
        _val_f1 = _val_f1[0]*0.2+_val_f1[1]*0.2+_val_f1[2]*0.6
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return
 
 
metrics = Metrics()


def model_train(model,x_train,y_train,batch_size,epochs,x_test,y_test, metrics):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_test, y_test),callbacks=[metrics])
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss",score[0])


def multiple_file_read(filePath,num,link_attr, link_connection, inv_link):
    path_list = os.listdir(filePath)
    for i in range(num):
        print(i)
        path = path_list[i]
        df = txtread(path)
        if i == 0:
            combined_feature, label = get_feature(path)
            x_train = get_link_feature(df, ink_attr, link_connection, inv_link, combined_feature)
        if i == 0:
            combined_feature, a = get_feature(path)
            label = np.concatenate((label,a),axis=0)
            total_feature_a = get_link_feature(df, ink_attr, link_connection, inv_link, combined_feature)
            x_train = np.concatenate((x_train,total_feature_a),axis=0)
    return x_train,label
