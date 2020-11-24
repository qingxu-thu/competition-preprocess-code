
# !pip install -U feather-format
# !python3 -m pip install paddlepaddle-gpu==2.0.0b0 -i https://mirror.baidu.com/pypi/simple
import pandas as pd
import os
import feather
from tqdm import tqdm
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm
from paddle.fluid.layers import dynamic_lstm
from paddle.fluid import layers
import joblib
import paddle
from sklearn.metrics import f1_score
paddle.disable_static()
print(paddle.__version__)
def txt2feather(path):
    ## 将txt文件转为feather文件，读取速度比较快
    ## eg:
    ## for p in tqdm(os.listdir('train/traffic/')):
    ##    txt2feather(p)
    df_0702 = pd.read_csv(f'train/traffic/{path}', sep=';', names=['linkid_' 'label_' 'current_slice_id_' 'future_slice_id', 'recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4'])
    df_test = pd.DataFrame(columns=['linkid', 'label', 'current_slice_id', 'future_slice_id'])
    df_test[['linkid', 'label', 'current_slice_id', 'future_slice_id']] = df_0702.iloc[:, 0].str.split(' ', expand=True)
    df_test = df_test.astype('int')
    df_test[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']] = df_0702[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']]
    df_test.to_feather(f'train/feather/{path[:-4]}.feather')
    return df_test
def get_feature(df):
    
    ##将recent_feature, history_feature1-4,合并并且储存成lstm所能使用格式。
    ##输入df
    ##输出lable，以及(-1, 5, 20)的numpy数组。

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
##将test集转化为lstm输入格式
df_0702 = pd.read_csv('test.txt', sep=';', names=['linkid_' 'label_' 'current_slice_id_' 'future_slice_id', 'recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4'])
df_test = pd.DataFrame(columns=['linkid', 'label', 'current_slice_id', 'future_slice_id'])
df_test[['linkid', 'label', 'current_slice_id', 'future_slice_id']] = df_0702.iloc[:, 0].str.split(' ', expand=True)
df_test = df_test.astype('int')
df_test[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']] = df_0702[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']]
df_test.to_feather('test.feather')
test = feather.read_dataframe('test.feather')
test_x, test_y = get_feature(test)
joblib.dump(test_x, 'test_x.pkl')
joblib.dump(test_y, 'test_y.pkl'
##将train集中7月1日当成训练集，7月2日当成验证集，转化为lstm输入格式
train_data = txt2feather('train/traffic/20190701.txt')
train_x, train_y = get_feature(train_data)
joblib.dump(train_x, '20190701_x.pkl')
joblib.dump(train_y, '20190701_y.pkl')

valid_data = txt2feather('train/traffic/20190702.txt')
valid_x, valid_y = get_feature(valid_data)
joblib.dump(valid_x, '20190702_x.pkl')
joblib.dump(valid_y, '20190702_y.pkl')
def data_loader(x_data=None, y_data=None, batch_size=1024):
    def reader():
        batch_data = []
        batch_labels = []
        for i in range(x_data.shape[0]):
            
            batch_labels.append(y_data[i])
            batch_data.append(x_data[i])

            if len(batch_data) == batch_size:
                batch_data = np.array(batch_data).astype('float32')
                batch_labels = np.array(batch_labels).astype('int')
                yield batch_data, batch_labels
                batch_data = []
                batch_labels = []
        if len(batch_data) > 0:
            batch_data = np.array(batch_data).astype('float32')
            batch_labels = np.array(batch_labels).astype('int')
            yield batch_data, batch_labels
            batch_data = []
            batch_labels = []
    return reader
class base_model(fluid.dygraph.Layer):
    def __init__(self, classes_num: int):
        super().__init__()
        self.hidden_size = 128
        self.batchNorm1d = paddle.nn.BatchNorm1d(5)
        self.lstm   = paddle.nn.LSTM(input_size=20, hidden_size=self.hidden_size, direction="bidirectional")
        
        self.avgpool1d = paddle.nn.AvgPool1d(kernel_size=self.hidden_size*2, stride=self.hidden_size*2)
        self.maxpool1d = paddle.nn.MaxPool1d(kernel_size=self.hidden_size*2, stride=self.hidden_size*2)


    def forward(self, input):
        #input:（batch_size, max_len, dim)
        
        x = self.batchNorm1d(input)
        x.stop_gradient = True

        rnn_out = self.lstm(x)[0]
        mean_out = self.avgpool1d(rnn_out)
        max_out = self.maxpool1d(rnn_out)
        r_shape = (mean_out.shape[0], mean_out.shape[1])
        mean_pool_out = layers.reshape(mean_out, shape=r_shape)
        max_pool_out = layers.reshape(max_out, shape=r_shape)
        add_output = mean_pool_out + max_pool_out
        concat_output = layers.concat((mean_pool_out, max_pool_out), axis=1)

        output = layers.fc(concat_output, size=5)
        return output
if __name__ == '__main__':
    # 创建模型
    # with fluid.dygraph.guard():
    program = fluid.default_main_program()
    program.random_seed = 2020
    model = base_model(4)
    print('start training ... {} kind'.format(4))
    model.train()
    epoch_num = 30
    # 定义优化器
    opt = fluid.optimizer.Adam(learning_rate=0.001, parameter_list=model.parameters())
    # 定义数据读取器，训练数据读取器和验证数据读取器
    x = joblib.load('train/preprocess_file/20190701_x.pkl')
    y = joblib.load('train/preprocess_file/20190701_y.pkl')
    val_x = joblib.load('train/preprocess_file/20190702_x.pkl')
    val_y = joblib.load('train/preprocess_file/20190702_y.pkl')
    train_loader = data_loader(x, y, 1024)
    valid_loader = data_loader(val_x, val_y, 1024)

    best_acc = 0
    valid_acc = 0

    print('start training ... {} kind'.format(4))
    for epoch in range(epoch_num):
        all_loss = 0
        model.train()
        
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            x = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            label = paddle.fluid.one_hot(label, depth=5)
            # 运行模型前向计算，得到预测值
            logits = model(x)
            # 进行loss计算
            softmax_logits = fluid.layers.softmax(logits)
            loss = fluid.layers.cross_entropy(softmax_logits, label, soft_label=True)
            avg_loss = fluid.layers.mean(loss)
            all_loss += avg_loss.numpy()
            avg_l = all_loss/(batch_id + 1)
            if(batch_id % 100 == 0):
                print("epoch: {}, batch_id: {}, loss is: {}, valid acc is: {}".format(epoch, batch_id, avg_loss.numpy(), valid_acc))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
            # break
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            x_data = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(x_data)
            # 计算sigmoid后的预测概率，进行loss计算
            pred = fluid.layers.softmax(logits)

            scores = f1_score(y_true=pred.numpy().argmax(axis=1), y_pred=y_data, average=None)

            scores = scores[0]*0.2 + scores[1]*0.2 + scores[2]*0.6
            accuracies.append(scores)
        valid_acc = np.mean(accuracies)