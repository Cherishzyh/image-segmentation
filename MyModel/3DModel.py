import numpy as np

from keras.models import Model
from keras.layers import Input, BatchNormalization, Concatenate, PReLU
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from keras.utils import plot_model


def Encoding(inputs, filters=64, blocks=4):
    outputs = []
    x = inputs
    for index in range(blocks):
        x = Conv3D(filters * np.power(2, index), kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Conv3D(filters * np.power(2, index), kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        if index != blocks - 1:
            outputs.append(x)
            x = MaxPooling3D(pool_size=(2, 2, 1))(x)

    return x, outputs


def Decoding_Deep_Supervision(inputs_1, inputs_2, filters=64, blocks=4, channel=3):
    x = inputs_1
    output_list = []
    for index in np.arange(blocks - 2, -1, -1):
        x = Conv3DTranspose(filters * np.power(2, index), kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)
        x = Concatenate(axis=4)([x, inputs_2[index]])

        x = Conv3D(filters * np.power(2, index), kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Conv3D(filters * np.power(2, index), kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        output = Conv3D(channel, (1, 1, 1), activation='softmax')(x)
        output_list.append(output)

    output_list.reverse()

    return output_list


def DSUNet(input_shape, filters=64, blocks=4):
    inputs = Input(input_shape)

    x1, EncodingList = Encoding(inputs, filters, blocks)

    x2 = Decoding_Deep_Supervision(x1, EncodingList, filters, blocks)

    model = Model(inputs, x2)
    return model


if __name__ == '__main__':
    model = DSUNet(input_shape=(240, 240, 3, 1), filters=64, blocks=4)
    model.summary()
    plot_model(model, to_file=r'C:\Users\ZhangYihong\Desktop\model.png', show_shapes=True, )
