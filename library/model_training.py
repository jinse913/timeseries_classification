#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[32]:


from tensorflow.keras.callbacks import ModelCheckpoint


# In[1]:


from library.load_data_function import load_data
from library.timeseries_model import *


# In[ ]:


def plot_history(history):
    print(str(history))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss', fontsize = 20)
    plt.ylabel('loss', fontsize = 15)
    plt.xlabel('epoch', fontsize = 15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(['train', 'valid'], fontsize = 10)
    plt.show() 

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy', fontsize = 20)
    plt.ylabel('accuracy', fontsize = 15)
    plt.xlabel('epoch', fontsize = 15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(['train', 'valid'], fontsize = 10)
    plt.show() 


# In[39]:


def create_rpm(time_series):
    """
    time_series: (timesteps, channels) 형태의 시계열 데이터를 입력으로 받음
    """
    timesteps, channels = time_series.shape  # time_series가 (timesteps, channels) 형태라고 가정
    rpm = np.zeros((timesteps, timesteps, channels))  # 각 채널에 대해 별도의 RPM 계산

    for c in range(channels):
        for i in range(timesteps):
            for j in range(timesteps):
                # 각 채널에 대해 상대적 차이를 계산
                rpm[i, j, c] = abs(time_series[i, c] - time_series[j, c])
    
    return rpm


def model_training(model_name,train_X, train_Y, valid_X, valid_Y, Epochs, Batch_size):
    now = dt.datetime.now()

    MODEL_SAVE_FOLDER_PATH = "../Model/"+str(model_name)+'_'+str(now.year)+str(now.month)+str(now.day)+'_'+str(now.hour)+str(now.minute)+str(now.second)+'/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + 'epoch_{epoch:02d}_{val_loss:.4f}.h5'
    cd_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=0, save_best_only=True)
        
    
    if model_name == 'MC-DCNN':
        train_split = np.split(train_X, indices_or_sections=6, axis=2)
        valid_split = np.split(valid_X, indices_or_sections=6, axis=2)
        unique = len(set(train_Y))

        model = create_mc_dcnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_split, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_split, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'MC-CNN':
        unique = len(set(train_Y))

        model = create_mc_cnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history


    elif model_name == 'FCN':
        unique = len(set(train_Y))

        model = create_mc_cnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'ResNet':
        unique = len(set(train_Y))

        model = create_resnet(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'Res-CNN':
        unique = len(set(train_Y))

        model = create_res_cnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'DCNNs':
        unique = len(set(train_Y))

        model = create_dcnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'Disjoint-CNN':
        unique = len(set(train_Y))

        model = create_disjoint_cnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'RPMCNN':
        unique = len(set(train_Y))

        train_rpm = np.array([create_rpm(ts) for ts in tqdm(train_X)])
        train_rpm = np.expand_dims(train_rpm, axis=-1)  # Add channel dimension
        valid_rpm = np.array([create_rpm(ts) for ts in tqdm(valid_X)])
        valid_rpm = np.expand_dims(valid_rpm, axis=-1)

        model = create_rpmcnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_rpm, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_rpm, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'MCNN':
        unique = len(set(train_Y))

        model = create_mcnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 't-LeNet':
        unique = len(set(train_Y))

        model = create_tlenet(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'MVCNN':
        unique = len(set(train_Y))

        model = create_mvcnn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'InceptionTime':
        unique = len(set(train_Y))

        model = create_inceptiontime(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'EEG-Inception':
        unique = len(set(train_Y))

        model = create_eeg_inception(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'Inception-FCN':
        unique = len(set(train_Y))

        model = create_inception_fcn(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    
    elif model_name == 'LITE':
        unique = len(set(train_Y))

        model = create_lite(train_X, unique)

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
        history = model.fit(train_X, train_Y, epochs=Epochs, batch_size=Batch_size, validation_data = (valid_X, valid_Y), callbacks=[cd_checkpoint])

        plot_history(history)

        return model, history
    


# In[ ]:




