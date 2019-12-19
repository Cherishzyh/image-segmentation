import os

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

from CNNModel.Training.Generate import MyImageMultiInImageOut2D
from MeDIT.DataAugmentor import random_2d_augment
from CNNModel.Utility.SaveAndLoad import SaveModel


from MyModel.AttentionUNet import AttenNet
from ImageSegmentation.Train.LossFunction import FocalLoss
from Visualization.LossTendency import show_train_history

from FilePath.Filepath import train_folder, validation_folder, store_folder


def ThreeDTrain(model_folder):
    input_shape = [240, 240, 1]
    image_shape = [240, 240]
    batch_size = 16

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    number_training = len(os.listdir(train_folder))
    number_validation = len(os.listdir(validation_folder))

    # Generate 改过
    train_generator = MyImageMultiInImageOut2D(train_folder, image_shape, batch_size=batch_size, augment_param=random_2d_augment)
    validation_generator = MyImageMultiInImageOut2D(validation_folder, image_shape, batch_size=batch_size, augment_param=random_2d_augment)

    # MyModel
    model = AttenNet(input_shape)
    SaveModel(model, model_folder)

    # callbacks
    # 步长，初始设置为1e-3，如果validation loss 连续20代不下降，则步长变为原来的0.5.
    # EarlyStop，如果validation loss 连续100代不下降，则终止
    # CheckPoint，只保存最优模型的权重（对应Validation Loss最小的情况）
    # callbacks = [xxxx, xxxx, xxxx]
    callbacks = [
        TensorBoard(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min'),
        EarlyStopping(monitor='val_loss', patience=100, mode='min'),
        ModelCheckpoint(filepath=os.path.join(model_folder, 'weights.h5'), monitor='val_loss',
                        save_best_only=True, mode='min', period=1)
    ]

    model.compile(loss=FocalLoss, optimizer=Adam(0.001), metrics=['acc'], loss_weights=[0.6, 0.3, 0.1])

    history = model.fit_generator(train_generator, steps_per_epoch=number_training // batch_size, epochs=1000, verbose=1,
                                  validation_data=validation_generator, validation_steps=number_validation // batch_size,
                                  callbacks=callbacks)

    # model.save_weights(os.path.join(model_folder, 'last_weights.h5'))
    show_train_history(history, 'loss', 'val_loss', store_path=store_folder)


def main():
    ThreeDTrain(store_folder)


if __name__ == '__main__':
    main()








