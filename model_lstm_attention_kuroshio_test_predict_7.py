import numpy as np
import matplotlib
from tensorflow.keras.callbacks import LearningRateScheduler
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
from keras.models import Sequential, load_model, Model  # load_weights,
from keras import Input
from keras.layers import Dense, LSTM, Activation, Dropout, ConvLSTM2D, BatchNormalization, Conv3D, Lambda, Dot, Multiply, Add, Concatenate, Reshape
from sklearn.metrics import mean_squared_error
import scipy.io as sio
import tensorflow as tf
from keras.metrics import binary_crossentropy, mse
import keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
Batch_size = 10
Epochs = 60
TIMESTEPS = 21
OUTPUTDIM = 7
last_dim = 1


def create_dataset_train(dataset, TIMESTEPS, OUTPUTDIM):
    dataX, dataY = [], []
    dim = dataset.shape
    times = 1

    for i in range(0, int(dim[2]) - TIMESTEPS - OUTPUTDIM - 1 - 1):
        current_a = dataset[:, :, i:i + TIMESTEPS]
        current_a = current_a.reshape(TIMESTEPS,dim[0], dim[1])
        num = 0
        for j in range(i+1, i + TIMESTEPS+1):
            current_a[num, :, :] = dataset[:, :, j]
            num = num + 1
        current_a = tf.expand_dims(current_a, 3)

        current_b = dataset[:, :, i:i + OUTPUTDIM]
        current_b = current_b.reshape(OUTPUTDIM, dim[0], dim[1])
        num = 0
        for j in range((i + TIMESTEPS) + 1, (i + TIMESTEPS + OUTPUTDIM) + 1):
            current_b[num, :, :] = dataset[:, :, j]
            num = num + 1
        current_b = tf.expand_dims(current_b, 3)
        dataX.append(current_a)
        dataY.append(current_b)

    return np.array(dataX), np.array(dataY)


def create_dataset_test(dataset, TIMESTEPS, OUTPUTDIM):
    dataX, dataY = [], []
    dim = dataset.shape
    for i in range(0, int(dim[2]) - TIMESTEPS - OUTPUTDIM - 1):
        current_a = dataset[:, :, i:i + TIMESTEPS]
        current_a = current_a.reshape(TIMESTEPS, dim[0], dim[1])
        num = 0
        for j in range(i, i + TIMESTEPS):
            current_a[num,:,:] = dataset[:, :, j]
            num = num+1
        current_a = tf.expand_dims(current_a, 3)
        current_b = dataset[:, :, i:i + OUTPUTDIM]
        current_b = current_b.reshape(OUTPUTDIM, dim[0], dim[1])
        num = 0
        for j in range(i + TIMESTEPS, i + TIMESTEPS + OUTPUTDIM):
            current_b[num,:,:] = dataset[:, :, j]
            num = num+1
        current_b = tf.expand_dims(current_b, 3)
        dataX.append(current_a)
        dataY.append(current_b)

    return np.array(dataX), np.array(dataY)


def reshape_dataset(tempb, dim0, dim1):
    a = []
    b = []
    for j in range(dim0):
        for k in range(dim1):
            a.append(tempb[k, j])
        b.append(a)
    return np.array(b)


def reshape_y_hat(y_hat, dim):
    i = 0
    tmp_y = []
    while i < len(y_hat):
        t = 0

        while t < (y_hat.shape[1]):
            tmp = y_hat[i, t:t + dim]
            t = t + dim
            tmp_y.append(tmp)
        i = i + 1

    re_y = np.array(tmp_y, dtype='float32')
    return re_y



def vae_loss(y_true, y_pred):

    xent_loss = tf.square(y_true - y_pred)
    return xent_loss


def train_model(train_X, train_Y, Epochs, Batch_size, save_folder):
    # 设计网络0

    m, n, p, q, d = train_X.shape
    filter_number = 10

    # convolution part  
    model_input = Input(shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3], train_X.shape[4]))

    x_2 = ConvLSTM2D(filters=filter_number, kernel_size=(3, 3), input_shape=(None, p, q, d),
                   padding='same', activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
                   return_sequences=True)(model_input)
    x_2 = ConvLSTM2D(filters=filter_number, kernel_size=(1, 1), padding='same',
                   activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
                   return_sequences=True)(x_2)
    x_3 = ConvLSTM2D(filters=filter_number, kernel_size=(5, 5), input_shape=(None, p, q, d),
                   padding='same', activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
                   return_sequences=True)(model_input)
    x_3 = ConvLSTM2D(filters=filter_number, kernel_size=(3, 3), padding='same',
                   activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
                   return_sequences=True)(x_3)
    x_4 = ConvLSTM2D(filters=filter_number, kernel_size=(9, 9), input_shape=(None, p, q, d),
                   padding='same', activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
                   return_sequences=True)(model_input)
    x_4 = ConvLSTM2D(filters=filter_number, kernel_size=(5, 5), padding='same',
                   activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
                   return_sequences=True)(x_4)

    x_5 = Conv3D(filters=filter_number, kernel_size=(3, 3, 3), input_shape=(None, p, q, d), activation='tanh', padding='same',data_format='channels_last')(model_input)

    # this is attention part 
    units = 1
    inputs = Concatenate(axis=-1)([x_2, x_3, x_4])
    inputs = Dense(units, use_bias=True, activation='tanh')(inputs)

    units = 1
    x_2_x_1_part_1 = Dense(units, use_bias=True, activation='sigmoid')(inputs)
    x_2_x_1_part_2 = Dense(units, use_bias=True, activation='tanh')(inputs)
    part_3 = Dense(units, use_bias=True, activation='sigmoid')(inputs)
    x_2_x_1 = Multiply()([x_2_x_1_part_1, x_2_x_1_part_2])
    x_2_x_1_x_3 = Multiply()([x_2_x_1, part_3])


    inputs = Dense(units, use_bias=True, activation='tanh')(x_2_x_1_x_3)

    w_1_x_t = Lambda(lambda x: x[:, TIMESTEPS-OUTPUTDIM:TIMESTEPS, :, :, :], input_shape=(None, p, q, d))(model_input)
    w_2_h_t_1 = Lambda(lambda x: x[:, TIMESTEPS-OUTPUTDIM:TIMESTEPS, :, :, :])(inputs)
    x_2_x_1_3 = Lambda(lambda x: x[:, TIMESTEPS-OUTPUTDIM:TIMESTEPS, :, :, :])(x_5)
    i_t_init = Concatenate(axis=-1)([w_1_x_t, w_2_h_t_1])
    i_t_part_1 = Dense(units, use_bias=True, activation='sigmoid')(i_t_init)
    i_t_part_2 = Dense(units, use_bias=True, activation='tanh')(i_t_init)
    f_t = Dense(units, use_bias=True, activation='sigmoid')(i_t_init)


    o_t = Dense(units, use_bias=True, activation='tanh')(i_t_init)
    m_t_part_1 = Multiply()([x_2_x_1_3, f_t])
    m_t_part_2 = Multiply()([i_t_part_1, i_t_part_2])
    m_t_init = Add()([m_t_part_1, m_t_part_2])
    out_t = Multiply()([m_t_init, o_t])
    output = Lambda(lambda x: x[:, :, :, :, 9:10])(out_t)
    model = Model(model_input, output)

    print('Shape of model {model.summary()}')
    model.compile(loss=vae_loss, optimizer='adam', metrics=['acc'])
    model.summary()

    history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, callbacks=[lr_scheduler], validation_split=0.1)

    return model, history

def cosine_annealing(epoch,lr):
    T_max = 2
    eta_min = 0.00001
    eta_max = 0.01

    return eta_min + (eta_max - eta_min) * (1+math.cos(math.pi*epoch/T_max))/2

lr_scheduler = LearningRateScheduler(cosine_annealing)

os.environ["CUDA_DEVICES_ORDER"] = 'PCI_BUS_IS'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
f = open('results_kuroshio_test_compare_7.txt', 'a')
save_folder = 'saved_model_kuroshio_test_compare_7'
filenames = os.listdir('data_path')
for filename in filenames:
    Y = sio.loadmat('data_path/' + filename)
    rMSE_total = 0
    acc_total = 0
    f.write('10-->7 hidden=6, l_fc=1[7], l_r=1 \n')
    f.write('filename' + 'point' + ' MSE ' + ' rMSE ' + ' PCC ' + ' ACC ' + '\n')
    i = 1

    f.write(str(i) + ':')

    # 加载数据集
    x = Y['data_norm']
    dataset = x
    # 整数编码
    dataset = dataset.astype('float32')
    
    values = dataset


    n_train_hours = 4300  
    n_test_hours = 1000 + 4300  
    train = values[:, :, 0:n_train_hours]
    test = values[:, :, n_train_hours:n_test_hours]
    # 分为输入输出

    train_X, train_Y = create_dataset_train(train, TIMESTEPS, OUTPUTDIM)
    test_X, test_Y = create_dataset_test(test, TIMESTEPS, OUTPUTDIM)

    # 重塑成3D形状 [样例, 时间步, 特征]
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    # 绘制历史数据
    model, history = train_model(train_X, train_Y, Epochs, Batch_size, save_folder)
    # 做出预测
    # testPredict = model.predict(test_X)
    the_last_name = filename.rfind('.mat')
    # save model
    new_filename = filename[:the_last_name]
    model.save(save_folder + '/' + new_filename + 'compare_1.h5')

    #model.save_weights(save_folder + '/' + save_folder + '.h5')

    print(model.evaluate(test_X, test_Y, batch_size=Batch_size))
    y_hat = model.predict(test_X)
    
    inv_yhat = y_hat[:,:,:,:,0]
    inv_testY = test_Y[:,:,:,:,0]

  

    testY_ = inv_testY
    testPredict_ = inv_yhat
    sio.savemat(save_folder + '/' + new_filename + '_predict.mat',
                {'reshape_testPredict_': testPredict_, 'reshape_testY_': testY_})

    # plt.show()
    s_1, s_2, s_3, s_4 = testPredict_.shape
    # s_1, s_2, s_3 = testPredict_.shape
    testPredict_ = testPredict_.reshape(s_1, s_2 * s_3 * s_4)
    testY_ = testY_.reshape(s_1, s_2 * s_3 * s_4)
    test_b = tf.math.reduce_sum(testPredict_, axis=1)
    test_a = tf.math.reduce_sum(testY_, axis=1)
    fig, ax = plt.subplots(1)
    plot_test, = ax.plot(test_a)
    plot_predicted, = ax.plot(test_b)
    plt.title('SST Predictions')
    plt.legend([plot_predicted, plot_test], ['predicted', 'true value'])
    plt.savefig(save_folder + '/' + new_filename + '_predict')


    MSE = mean_squared_error(testPredict_, testY_)
    print("MSE: %f" % MSE)
    rMSE = math.sqrt(MSE)
    print('rMSE:%f' % rMSE)
    pcc = np.corrcoef(testPredict_, testY_, rowvar=0)[0, 1]
    print("PCC: %f" % pcc)
    acc = 1 - np.mean((np.abs(testPredict_ - testY_)) / (testY_ + testPredict_))
    print("ACC: %f" % acc)
    # sum
    rMSE_total = rMSE_total + rMSE
    acc_total = acc_total + acc
    # training epoch
    fig, ax = plt.subplots(1)
    loss, = ax.plot(history.history["loss"])
    val_loss, = ax.plot(history.history["val_loss"])
    plt.title('training process')
    plt.legend([loss, val_loss], ['loss', 'val loss'])
    plt.savefig(save_folder + '/' + new_filename + '_train')
    #    plt.show()

    # write to file
    f.write(filename + str(MSE) + ' ' + str(rMSE) + ' ' + str(pcc) + ' ' + str(acc) + '\n')
    rMSE_ave = rMSE_total
    acc_ave = acc_total
    f.write('\n average rMSE   ACC \n')
    f.write(str(rMSE_ave) + ' ' + str(acc_ave))

f.close()