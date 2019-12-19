import numpy as np

from keras.models import Model
from keras.layers import Input, BatchNormalization, Concatenate, PReLU
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose


def Encoding(inputs, filters=64, blocks=4):
    outputs = []
    x = inputs
    for index in range(blocks):
        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        if index != blocks - 1:
            outputs.append(x)
            x = MaxPooling2D((2, 2))(x)

    return x, outputs


def Decoding_Deep_Supervision(inputs_1, inputs_2, filters=64, blocks=4, channel=3):
    x = inputs_1
    output_list = []
    for index in np.arange(blocks - 2, -1, -1):
        x = Conv2DTranspose(filters * np.power(2, index), kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, inputs_2[index]])

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        output = Conv2D(channel, (1, 1), activation='softmax')(x)
        output_list.append(output)

    output_list.reverse()

    return output_list


def DSUNet(input_shape, filters=64, blocks=4):
    inputs = Input(input_shape)

    x1, EncodingList = Encoding(inputs, filters, blocks)

    x2 = Decoding_Deep_Supervision(x1, EncodingList, filters, blocks)

    model = Model(inputs, x2)
    return model





