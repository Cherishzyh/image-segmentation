import numpy as np

from keras.models import Model
from keras.layers import Input, BatchNormalization, Concatenate, PReLU, Gr
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.utils import plot_model


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


def Decoding_Deep_Supervision(inputs_1, EncodingList1, EncodingList2, EncodingList3, filters=64, blocks=4, channel=3):
    x = inputs_1
    output_list = []
    for index in np.arange(blocks - 2, -1, -1):
        x = Conv2DTranspose(filters * np.power(2, index), kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, EncodingList1[index], EncodingList2[index], EncodingList3[index]])

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


def DSUNet(input_shape, filters=64, blocks=4, channel=3):
    input1 = Input(input_shape)
    input2 = Input(input_shape)
    input3 = Input(input_shape)

    x1, EncodingList1 = Encoding(input1, filters, blocks)
    x2, EncodingList2 = Encoding(input2, filters, blocks)
    x3, EncodingList3 = Encoding(input3, filters, blocks)

    x = Concatenate(axis=3)([x1, x2, x3])

    output = Decoding_Deep_Supervision(x, EncodingList1, EncodingList2, EncodingList3, filters, blocks, channel)

    model = Model([input1, input2, input3], output)
    return model



def main():
    model = DSUNet(input_shape=(240, 240, 3), channel=3)
    model.summary()
    plot_model(model, to_file=r'C:\Users\ZhangYihong\Desktop\model.png', show_shapes=True)


if __name__ == '__main__':
    main()


