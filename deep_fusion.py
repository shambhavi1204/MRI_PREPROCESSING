import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, Activation

def dense_block(x, filters):
    
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters, kernel_size=(3, 3), padding='same')(concatenate([x, conv1]))
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(filters, kernel_size=(3, 3), padding='same')(concatenate([x, conv1, conv2]))
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = Conv2D(filters, kernel_size=(3, 3), padding='same')(concatenate([x, conv1, conv2, conv3]))
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    return concatenate([x, conv1, conv2, conv3, conv4])

def densefuse_model(input_shape):
   
    input_visible = tf.keras.Input(shape=input_shape)
    input_infrared = tf.keras.Input(shape=input_shape)

    encoder_conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(input_visible)
    encoder_conv1 = BatchNormalization()(encoder_conv1)
    encoder_conv1 = Activation('relu')(encoder_conv1)

    encoder_pool1 = MaxPooling2D(pool_size=(2, 2))(encoder_conv1)

    # Dense Blocks
    dense_block1 = dense_block(encoder_pool1, 64)
    dense_block2 = dense_block(dense_block1, 128)
    dense_block3 = dense_block(dense_block2, 256)

    # Decoder
    decoder_upsampling1 = UpSampling2D(size=(2, 2))(dense_block3)
    decoder_conv1 = Conv2D(128, kernel_size=(3, 3), padding='same')(decoder_upsampling1)
    decoder_conv1 = BatchNormalization()(decoder_conv1)
    decoder_conv1 = Activation('relu')(decoder_conv1)

    decoder_upsampling2 = UpSampling2D(size=(2, 2))(decoder_conv1)
    decoder_conv2 = Conv2D(64, kernel_size=(3, 3), padding='same')(decoder_upsampling2)
    decoder_conv2 = BatchNormalization()(decoder_conv2)
    decoder_conv2 = Activation('relu')(decoder_conv2)

    # Output
    output = Conv2D(3, kernel_size=(1, 1), activation='sigmoid')(decoder_conv2)

    
    model = tf.keras.Model(inputs=[input_visible, input_infrared], outputs=output)

    return model


DWI_folder = 'path/to/visible/images'
ADC_folder = 'path/to/infrared/images'
output_folder = 'path/to/output/folder'


os.makedirs(output_folder, exist_ok=True)

# Load DenseFuse model
input_shape = (None, None, 3)
model = densefuse_model(input_shape)


# Get the list of image files in the input folders
DWI_files = os.listdir(DWI_folder)
ADC_files = os.listdir(ADC_folder)


for DWI_file, ADC_file in zip(DWI_files, ADC_files):
    
    DWI_image = cv2.imread(os.path.join(DWI_folder, DWI_file))
    ADC_image = cv2.imread(os.path.join(ADC_folder, ADC_file))

   
    fused_image = model.predict([DWI_image, ADC_image])

   
    output_file = os.path.splitext(DWI_file)[0] + '_fused.jpg'
    cv2.imwrite(os.path.join(output_folder, output_file), fused_image)

    print(f"Fusion complete for {DWI_file} and {ADC_file}. Fused image saved as {output_file}")
