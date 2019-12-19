from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from Train.LossFunction import dice_coef_loss


def DSN(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv2D(8, (3, 3), padding='same', activation='selu', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(8, (3, 3), padding='same', activation='selu', kernel_initializer='he_normal')(conv1)
    print("conv1 shape:", conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:", pool1.shape)
    convT1 = Conv2DTranspose(2, (2, 2), padding='valid', strides=(2, 2), activation='selu',
                             kernel_initializer='he_normal')(pool1)
    print("convT1 shape:", convT1.shape)
    out1 = Conv2D(1, 1, activation='softmax')(convT1)

    conv2 = Conv2D(16, (3, 3), padding='same', activation='selu', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, (3, 3), padding='same', activation='selu', kernel_initializer='he_normal')(conv2)
    print("conv2 shape:", conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:", pool2.shape)
    convT2 = Conv2DTranspose(2, (2, 2), padding='valid', strides=(2, 2), activation='selu',
                             kernel_initializer='he_normal')(pool2)
    convT2 = Conv2DTranspose(2, (2, 2), padding='valid', strides=(2, 2), activation='selu',
                             kernel_initializer='he_normal')(convT2)
    print("convT2 shape:", convT2.shape)
    out2 = Conv2D(1, 1, activation='softmax')(convT2)

    conv3 = Conv2D(32, (3, 3), padding='same', activation='selu', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, (3, 3), padding='same', activation='selu', kernel_initializer='he_normal')(conv3)
    print("conv3 shape:", conv3.shape)
    convT3 = Conv2DTranspose(2, (2, 2), padding='valid', strides=(2, 2), activation='selu',
                             kernel_initializer='he_normal')(conv3)
    convT3 = Conv2DTranspose(2, (2, 2), padding='valid', strides=(2, 2), activation='selu',
                             kernel_initializer='he_normal')(convT3)
    print("convT3 shape:", convT3.shape)
    out3 = Conv2D(1, 1, activation='sigmoid')(convT3)

    model = Model(input=inputs, output=[out3, out2, out1])
    adam = Adam(lr=0.0001)
    model.summary()

    model.compile(optimizer=adam, loss=dice_coef_loss, loss_weights=[0.6, 0.3, 0.1])

    return model


def train(self):
    print("loading data")
    imgs_train, label_train = self.load_train_data()
    print("loading data done")
    model = self.get_unet()
    print("got unet")

    # 保存的是模型和权重,
    model_checkpoint = ModelCheckpoint('seg_liver3D.h5', monitor='loss', verbose=0, save_best_only=True,
                                       save_weights_only=True, mode='min')
    print('Fitting model...')
    model.fit(imgs_train, [label_train, label_train, label_train], batch_size=2, epochs=15, verbose=1,
              callbacks=[model_checkpoint], validation_split=0.2, shuffle=True)

def main():
    model = DSN((496, 496, 3))
    model.summary()


if __name__ == '__main__':
    main()