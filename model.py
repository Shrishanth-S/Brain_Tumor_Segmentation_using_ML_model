# Import necessary libraries
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, 
                                    concatenate, GlobalAveragePooling2D, Reshape, 
                                    UpSampling2D, Multiply, Add, Activation)
from tensorflow.keras.models import Model

# -------------------------------
# Define the U-Net Model
# -------------------------------

def aspp_block(inputs, filters=128):
    atrous_rates = [6, 12, 18, 24]
    conv_1x1 = Conv2D(filters, kernel_size=1, padding="same", activation="relu")(inputs)
    
    atrous_convs = []
    for rate in atrous_rates:
        conv = Conv2D(filters, kernel_size=3, dilation_rate=rate, 
                     padding="same", activation="relu")(inputs)
        atrous_convs.append(conv)

    global_avg_pool = GlobalAveragePooling2D()(inputs)
    global_avg_pool = Reshape((1, 1, inputs.shape[-1]))(global_avg_pool)
    global_avg_pool = Conv2D(filters, kernel_size=1, activation="relu")(global_avg_pool)
    global_avg_pool = UpSampling2D(size=(inputs.shape[1], inputs.shape[2]), 
                         interpolation="bilinear")(global_avg_pool)

    concatenated = concatenate([conv_1x1] + atrous_convs + [global_avg_pool])
    output = Conv2D(filters, kernel_size=1, padding="same", activation="relu")(concatenated)
    return output

def attention_gate(input_tensor, gating_tensor, filters, kernel_size=(1, 1)):
    theta_x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(input_tensor)
    phi_g = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(gating_tensor)
    add_xg = Add()([theta_x, phi_g])
    relu_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, kernel_size=(1, 1), strides=1, padding='same')(relu_xg)
    psi = Activation('sigmoid')(psi)
    output = Multiply()([input_tensor, psi])
    return output

def unet_model(input_size=(128, 128, 4), num_classes=4):  # Note the 1 channel input
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck with ASPP
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = aspp_block(c5, filters=1024)

    # Decoder with attention gates
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    att6 = attention_gate(c4, u6, filters=512)
    u6 = concatenate([u6, att6])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    att7 = attention_gate(c3, u7, filters=256)
    u7 = concatenate([u7, att7])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    att8 = attention_gate(c2, u8, filters=128)
    u8 = concatenate([u8, att8])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    att9 = attention_gate(c1, u9, filters=64)
    u9 = concatenate([u9, att9])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model