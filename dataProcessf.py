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
    f = os.open(link_path)
    

    connection_matrix = get_connection_matrix(connect_feature)


def dict_map(df):




def get_connection_matrix(df):
    for item in df:
        for 


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
    model.summary()
    model.compile(loss='mse',optimizer=Adam())
    return model

