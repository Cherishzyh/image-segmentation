from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, BatchNormalization, Add, Activation, AveragePooling2D, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, concatenate

# some things before def
# for Google net
def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)

    return x


def Inception(x, nb_filter):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
    return x

# for Res_Net
def identity_block(inputs, num_filters, kernel_size=3,strides=1):
    x_shortcut = inputs

    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x


def conventional_block(inputs, num_filters, kernel_size, strides=1, activation='relu'):
    X_shortcut = Conv2D(num_filters, kernel_size=(1, 1), strides=2, padding='same')(inputs)

    X = Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same')(inputs)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(X)
    X = BatchNormalization()(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


# for U_net
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

# all kinds of net models


def AlexNet(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def GoogleNet(input_shape):
    input = Input(input_shape)
    # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))

    x = Conv2d_BN(input, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception(x, 64)  # 256
    x = Inception(x, 120)  # 480
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)  # 512
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)  # 528
    x = Inception(x, 208)  # 832
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)  # 1024
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(input, x)
    return model


def ResNet(input_shape):
    input = Input(input_shape)

    X = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = identity_block(X, num_filters=16, kernel_size=3, strides=1)
    X = identity_block(X, num_filters=16, kernel_size=3, strides=1)
    X = identity_block(X, num_filters=16, kernel_size=3, strides=1)

    X = conventional_block(X, num_filters=32, kernel_size=3, strides=1)
    X = identity_block(X, num_filters=32, kernel_size=3, strides=1)
    X = identity_block(X, num_filters=32, kernel_size=3, strides=1)

    X = conventional_block(X, num_filters=64, kernel_size=3, strides=1)
    X = identity_block(X, num_filters=64, kernel_size=3, strides=1)
    X = identity_block(X, num_filters=64, kernel_size=3, strides=1)

    X = AveragePooling2D(pool_size=2)(X)
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = Dense(1000, activation='relu')(X)
    X = Dense(10, activation='softmax')(X)

    model = Model(input, X)
    return model


def u_net(input_shape):

    inputs = Input(input_shape)

    a1 = Conv1(inputs, 16)
    p1 = MaxPooling2D((2, 2))(a1)
    a2 = Conv1(p1, 32)
    p2 = MaxPooling2D((2, 2))(a2)
    a3 = Conv1(p2, 64)
    p3 = MaxPooling2D((2, 2))(a3)
    a4 = Conv1(p3, 128)
    p4 = MaxPooling2D((2, 2))(a4)
    a5 = Conv1(p4, 256)
    b1 = Conv2(a5, a4, 128)
    b2 = Conv2(b1, a3, 64)
    b3 = Conv2(b2, a2, 32)
    b4 = Conv2(b3, a1, 16)

    outputs = Conv2D(1, (1, 1), activation='softmax')(b4)

    model = Model(inputs, outputs)
    return model


model = u_net(input_shape=(32, 32, 3))
print(model.summary())

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




