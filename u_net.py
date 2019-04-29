from keras.models import Model
from keras.layers import Input, BatchNormalization, Concatenate, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose


def Conv1(inputs, num_filters):
    x = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def Conv2(inputs_1, inputs_2, num_filters):
    x = Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(inputs_1)
    x = Concatenate(axis=3)([x, inputs_2])

    x = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def u_net(input_shape):

    inputs = Input(input_shape)

    a1 = Conv1(inputs, 16)
    p1 = MaxPooling2D((2, 2))(a1)
    a2 = Conv1(p1, 32)
    p2 = MaxPooling2D((2, 2))(a2)
    a3 = Conv1(p2, 64)
    b1 = Conv2(a3, a2, 32)
    b2 = Conv2(b1, a1, 16)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(b2)

    model = Model(inputs, outputs)
    return model


def test_model():
    model = u_net((180, 180, 1))
    model.summary()

# test_model()


