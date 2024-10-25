#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate,Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D, Activation, Add, AveragePooling1D, Multiply


# In[14]:


def create_mc_dcnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Inputs for each channel
    inputs = []
    conv_outputs = []

    for i in range(feats):
        # Independent convolution for each channel
        input_channel = Input(shape=(window_length, 1))
        inputs.append(input_channel)

        conv1 = Conv1D(filters=64, kernel_size=5, activation='relu')(input_channel)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(conv1)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        
        conv3 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv2)
        conv_outputs.append(Flatten()(conv3))

    # Concatenate feature maps from all channels
    merged = Concatenate()(conv_outputs)

    # Fully connected layer
    fc = Dense(128, activation='relu')(merged)
    fc = Dropout(0.5)(fc)

    # Output layer
    output = Dense(num_classes, activation='softmax')(fc)

    # Model creation
    model = Model(inputs=inputs, outputs=output)

    model.summary()
    
    return model


# In[15]:


def create_mc_cnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    input_layer = Input(shape=(window_length, feats))

    # 1st Convolutional layer (3-stage convolution)
    conv1 = Conv1D(filters=64, kernel_size=5, activation='relu')(input_layer)
    conv1 = MaxPooling1D(pool_size=2)(conv1)

    # 2nd Convolutional layer
    conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(conv1)
    conv2 = MaxPooling1D(pool_size=2)(conv2)

    # 3rd Convolutional layer
    conv3 = Conv1D(filters=256, kernel_size=3, activation='relu')(conv2)
    conv3 = MaxPooling1D(pool_size=2)(conv3)

    # Flattening the convolution output
    flat = Flatten()(conv3)

    # Fully connected layer
    fc = Dense(128, activation='relu')(flat)
    fc = Dropout(0.5)(fc)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(fc)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()
    
    return model


# In[16]:


def create_fcn(train_data, num_classes):
    # Input layer for time series data

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    
    input_layer = Input(shape=(window_length, feats))
    # 1st Convolutional layer
    conv1 = Conv1D(filters=128, kernel_size=8, activation='relu')(input_layer)

    # 2nd Convolutional layer
    conv2 = Conv1D(filters=256, kernel_size=5, activation='relu')(conv1)

    # 3rd Convolutional layer
    conv3 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv2)

    # Global Average Pooling instead of Fully Connected layer
    gap = GlobalAveragePooling1D()(conv3)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(gap)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)
    model.summary()
    
    return model


# In[17]:


def residual_block(x, filters, kernel_size):
    shortcut = x
    
    # First convolution
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # Second convolution
    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    
    # Add shortcut connection
    output = Add()([shortcut, conv2])
    output = Activation('relu')(output)
    
    return output

# Define ResNet model
def create_resnet(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for time series data
    input_layer = Input(shape=(window_length, feats))

    # Initial convolution layer
    conv = Conv1D(filters=64, kernel_size=7, padding='same')(input_layer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # 3 Residual blocks
    for _ in range(3):
        conv = residual_block(conv, filters=64, kernel_size=3)

    # Global Average Pooling layer
    gap = GlobalAveragePooling1D()(conv)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(gap)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)
    
    model.summary()
    
    return model


# In[18]:


def create_res_cnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for time series data
    input_layer = Input(shape=(window_length, feats))

    # 1 Residual block
    conv = Conv1D(filters=64, kernel_size=7, padding='same')(input_layer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = residual_block(conv, filters=64, kernel_size=3)

    # FCN structure (3 Convolutional layers)
    conv_fcn1 = Conv1D(filters=128, kernel_size=8, activation='relu')(conv)
    conv_fcn2 = Conv1D(filters=256, kernel_size=5, activation='relu')(conv_fcn1)
    conv_fcn3 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv_fcn2)

    # Global Average Pooling layer
    gap = GlobalAveragePooling1D()(conv_fcn3)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(gap)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()
    
    return model


# In[19]:


def create_dcnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for multivariate time series data
    input_layer = Input(shape=(window_length, feats))

    # 1st Dilated Convolutional layer (dilation rate = 1)
    conv1 = Conv1D(filters=64, kernel_size=3, dilation_rate=1, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # 2nd Dilated Convolutional layer (dilation rate = 2)
    conv2 = Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    # 3rd Dilated Convolutional layer (dilation rate = 4)
    conv3 = Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    # 4th Dilated Convolutional layer (dilation rate = 8)
    conv4 = Conv1D(filters=128, kernel_size=3, dilation_rate=8, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # Global Average Pooling layer
    gap = GlobalAveragePooling1D()(conv4)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(gap)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()
    
    return model


# In[20]:


def smoothing_layer(x):
    return AveragePooling1D(pool_size=2, strides=1, padding='same')(x)

# Define Down-sampling (Simple max pooling)
def down_sampling_layer(x):
    return AveragePooling1D(pool_size=2)(x)

# Define MCNN model
def create_mcnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for time series data
    input_layer = Input(shape=(window_length, feats))

    # Identity mapping (just forwarding the original input)
    smoothed_input = smoothing_layer(input_layer)

    # 1st Convolutional layer with smoothing
    conv1 = Conv1D(filters=64, kernel_size=3, padding='same')(smoothed_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # 2nd Convolutional layer with down-sampling
    conv2 = Conv1D(filters=128, kernel_size=3, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    down_sampled = down_sampling_layer(conv2)

    # Identity mapping needs to match down_sampled shape
    identity_mapping = Conv1D(filters=128, kernel_size=1, padding='same')(input_layer)
    identity_mapping = down_sampling_layer(identity_mapping)

    # Add Identity mapping (skip connection)
    combined = Add()([identity_mapping, down_sampled])

    # Flatten and fully connected layer
    flat = Flatten()(combined)
    fc = Dense(256, activation='relu')(flat)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(fc)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)
    
    model.summary()

    return model


# In[21]:


def temporal_conv_block(x, filters, kernel_size, dilation_rate=1):
    conv = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

# Spatial Convolution block (disjoint processing of channels)
def spatial_conv_block(x, filters, kernel_size):
    conv = []
    for i in range(x.shape[-1]):  # Apply convolution on each channel independently
        conv_channel = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x[:, :, i:i+1])
        conv_channel = BatchNormalization()(conv_channel)
        conv_channel = Activation('relu')(conv_channel)
        conv.append(conv_channel)
    # Concatenate across channels
    output = Concatenate(axis=-1)(conv)
    return output

# Define Disjoint-CNN model
def create_disjoint_cnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for multivariate time series data
    input_layer = Input(shape=(window_length, feats))

    # Temporal Convolution (applied to the full sequence)
    temporal_conv = temporal_conv_block(input_layer, filters=64, kernel_size=3, dilation_rate=1)
    temporal_conv = temporal_conv_block(temporal_conv, filters=128, kernel_size=3, dilation_rate=2)
    
    # Spatial Convolution (applied independently to each channel)
    spatial_conv = spatial_conv_block(input_layer, filters=128, kernel_size=3)

    # Merge Temporal and Spatial Conv results
    merged = Concatenate()([temporal_conv, spatial_conv])

    # Global Average Pooling layer
    gap = GlobalAveragePooling1D()(merged)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(gap)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()
    
    return model


# In[22]:


# Define RPMCNN model (VGGNet-based, 2-stage Conv)
def create_rpmcnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for RPM image data
    input_layer = Input(shape=(window_length, window_length, feats))

    # 1st Convolutional stage (Conv + MaxPooling)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 2nd Convolutional stage (Conv + MaxPooling)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Flatten and fully connected layers
    flat = Flatten()(conv2)
    fc1 = Dense(256, activation='relu')(flat)
    
    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(fc1)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()
    
    return model


# In[23]:


def create_tlenet(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for time series data
    input_layer = Input(shape=(window_length, feats))

    # 1st Convolutional layer with dilation
    conv1 = Conv1D(filters=64, kernel_size=5, dilation_rate=1, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # Squeeze: Reducing channels
    squeeze = Conv1D(filters=32, kernel_size=1, padding='same')(conv1)
    squeeze = BatchNormalization()(squeeze)
    squeeze = Activation('relu')(squeeze)

    # 2nd Convolutional layer with dilation
    conv2 = Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='same')(squeeze)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    # Global Average Pooling
    gap = AveragePooling1D(pool_size=2)(conv2)

    # Flatten and fully connected layer
    flat = Flatten()(gap)
    fc = Dense(256, activation='relu')(flat)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(fc)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()

    return model


# In[24]:


def inception_module_for_mvcnn(input_tensor, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    # 1x1 convolution branch
    conv1x1 = Conv1D(filters_1x1, kernel_size=1, padding='same')(input_tensor)
    conv1x1 = BatchNormalization()(conv1x1)
    conv1x1 = Activation('relu')(conv1x1)
    
    # 3x3 convolution branch
    conv3x3_reduce = Conv1D(filters_3x3_reduce, kernel_size=1, padding='same')(input_tensor)
    conv3x3_reduce = BatchNormalization()(conv3x3_reduce)
    conv3x3_reduce = Activation('relu')(conv3x3_reduce)
    
    conv3x3 = Conv1D(filters_3x3, kernel_size=3, padding='same')(conv3x3_reduce)
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Activation('relu')(conv3x3)
    
    # 5x5 convolution branch
    conv5x5_reduce = Conv1D(filters_5x5_reduce, kernel_size=1, padding='same')(input_tensor)
    conv5x5_reduce = BatchNormalization()(conv5x5_reduce)
    conv5x5_reduce = Activation('relu')(conv5x5_reduce)
    
    conv5x5 = Conv1D(filters_5x5, kernel_size=5, padding='same')(conv5x5_reduce)
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Activation('relu')(conv5x5)
    
    # Max Pooling branch
    maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    maxpool_proj = Conv1D(filters_pool_proj, kernel_size=1, padding='same')(maxpool)
    maxpool_proj = BatchNormalization()(maxpool_proj)
    maxpool_proj = Activation('relu')(maxpool_proj)
    
    # Concatenate all branches
    output = Concatenate()([conv1x1, conv3x3, conv5x5, maxpool_proj])
    
    return output

# Define MVCNN model
def create_mvcnn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    # Input layer for time series data
    input_layer = Input(shape=(window_length, feats))
    
    # 1st Inception module
    inception1 = inception_module_for_mvcnn(input_layer, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
                                  filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)
    
    # 2nd Inception module
    inception2 = inception_module_for_mvcnn(inception1, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                                  filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)
    
    # 3rd Inception module
    inception3 = inception_module_for_mvcnn(inception2, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
                                  filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)
    
    # 4th Inception module
    inception4 = inception_module_for_mvcnn(inception3, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
                                  filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    
    # Global Average Pooling and fully connected layer
    flat = Flatten()(inception4)
    fc = Dense(256, activation='relu')(flat)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(fc)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()

    return model


# In[25]:


def inception_module_for_inceptiontime(input_tensor, filters):
    # Branch 1: 1x1 convolution
    conv1 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # Branch 2: 1x1 -> 3x3 convolution
    conv3 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv1D(filters=filters, kernel_size=3, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    # Branch 3: 1x1 -> 5x5 convolution
    conv5 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv1D(filters=filters, kernel_size=5, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    # Branch 4: Max Pooling -> 1x1 convolution
    maxpool = Conv1D(filters=filters, kernel_size=3, padding='same', strides=1)(input_tensor)
    maxpool = BatchNormalization()(maxpool)
    maxpool = Activation('relu')(maxpool)

    # Concatenate all branches
    output = Concatenate()([conv1, conv3, conv5, maxpool])
    
    return output

# Define InceptionTime model (ensemble version)
def create_inceptiontime(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    input_layer = Input(shape=(window_length, feats))
    
    # Create 5 inception modules to simulate the ensemble effect (InceptionTime uses ensemble of multiple models)
    ensemble_outputs = []
    for _ in range(5):  # Ensemble of 5 models
        x = inception_module_for_inceptiontime(input_layer, filters=32)
        x = inception_module_for_inceptiontime(x, filters=32)
        x = inception_module_for_inceptiontime(x, filters=32)
        
        # Global Average Pooling
        gap = GlobalAveragePooling1D()(x)
        
        # Fully connected layer
        output = Dense(num_classes, activation='softmax')(gap)
        ensemble_outputs.append(output)

    # Combine outputs from the ensemble
    if len(ensemble_outputs) > 1:
        output = Add()(ensemble_outputs)
    else:
        output = ensemble_outputs[0]

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()
    
    return model


# In[26]:


def cross_branch_attention(branch_outputs):
    # Concatenate branch outputs
    concat = Concatenate()(branch_outputs)
    
    # Apply attention (softmax over concatenated branches)
    attention_weights = Dense(len(branch_outputs), activation='softmax')(concat)
    
    # Apply attention weights to branches
    attention_outputs = [Multiply()([branch, attention_weights]) for branch in branch_outputs]
    
    # Summing up the weighted branches
    output = Concatenate()(attention_outputs)
    
    return output

# Define Inception module (based on InceptionTime)
def inception_module_for_eeg(input_tensor, filters):
    # Branch 1: 1x1 convolution
    conv1 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # Branch 2: 1x1 -> 3x3 convolution
    conv3 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv1D(filters=filters, kernel_size=3, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    # Branch 3: 1x1 -> 5x5 convolution
    conv5 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv1D(filters=filters, kernel_size=5, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # Concatenate all branches
    output = Concatenate()([conv1, conv3, conv5])
    
    return output

# Define EEG-Inception model
def create_eeg_inception(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    input_layer = Input(shape=(window_length, feats))
    
    # First Inception module
    inception1 = inception_module_for_eeg(input_layer, filters=32)
    
    # Cross Branch Attention applied to the outputs of first Inception module
    attention1 = cross_branch_attention([inception1])
    
    # Second Inception module
    inception2 = inception_module_for_eeg(attention1, filters=32)
    
    # Cross Branch Attention applied to the outputs of second Inception module
    attention2 = cross_branch_attention([inception2])
    
    # Global Average Pooling and fully connected layer
    gap = GlobalAveragePooling1D()(attention2)
    fc = Dense(256, activation='relu')(gap)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(fc)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()

    return model


# In[27]:


def inception_module_for_inceptionfcn(input_tensor, filters):
    # Branch 1: 1x1 convolution
    conv1 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # Branch 2: 1x1 -> 3x3 convolution
    conv3 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv1D(filters=filters, kernel_size=3, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    # Branch 3: 1x1 -> 5x5 convolution
    conv5 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv1D(filters=filters, kernel_size=5, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # Concatenate all branches
    output = Concatenate()([conv1, conv3, conv5])
    
    return output

# Define Inception-FCN model
def create_inception_fcn(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]
    
    input_layer = Input(shape=(window_length, feats))
    
    # Inception modules (Inception Time part)
    x = inception_module_for_inceptionfcn(input_layer, filters=32)
    x = inception_module_for_inceptionfcn(x, filters=32)
    x = inception_module_for_inceptionfcn(x, filters=32)
    
    # FCN part: Global Average Pooling and fully connected layer
    gap = GlobalAveragePooling1D()(x)
    fc = Dense(256, activation='relu')(gap)

    # Output layer (for classification)
    output = Dense(num_classes, activation='softmax')(fc)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()

    return model


# In[38]:


# Define Inception module with custom filters and dilated convolutions
def inception_module_for_lite(input_tensor, filters):
    # Branch 1: 1x1 convolution (standard filter)
    conv1 = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # Branch 2: 3x3 convolution with dilation
    conv3 = Conv1D(filters=filters, kernel_size=3, padding='same', dilation_rate=2)(input_tensor)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    # Branch 3: 5x5 custom convolution with dilation
    conv5 = Conv1D(filters=filters, kernel_size=5, padding='same', dilation_rate=3)(input_tensor)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # Branch 4: Custom convolution with varying kernel size and dilation rate
    conv_custom = Conv1D(filters=filters, kernel_size=7, padding='same', dilation_rate=4)(input_tensor)
    conv_custom = BatchNormalization()(conv_custom)
    conv_custom = Activation('relu')(conv_custom)
    
    # Concatenate all branches
    output = Concatenate()([conv1, conv3, conv5, conv_custom])
    
    return output

# Define LITE model
def create_lite(train_data, num_classes):

    window_length = np.shape(train_data)[1]
    feats = np.shape(train_data)[2]

    input_layer = Input(shape=(window_length,feats))
    
    # Inception modules (LITE structure with custom filters)
    x = inception_module_for_lite(input_layer, filters=32)
    x = inception_module_for_lite(x, filters=32)
    x = inception_module_for_lite(x, filters=32)
    
    # Global Average Pooling and fully connected layer
    gap = GlobalAveragePooling1D()(x)
    output = Dense(num_classes, activation='softmax')(gap)

    # Model creation
    model = Model(inputs=input_layer, outputs=output)

    model.summary()

    return model


# In[ ]:




