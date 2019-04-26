from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from u_net import u_net
from data import GetData
from filepath import train_data_folder, validation_data_folder, store_folder
from loss_function import dice_coef_loss
from visualization import show_train_history
from saveandload import SaveModel
import os


training_data, training_label = GetData(train_data_folder)
validation_data, validation_label = GetData(validation_data_folder)

# training_data.shape = (number, row, col, channel)
input_shape = training_data.shape[1:]
model = u_net(input_shape)
print(model.summary())

SaveModel(model, store_folder)

# callbacks
# 步长，初始设置为1e-3，如果validation loss 连续20代不下降，则步长变为原来的0.5.
# EarlyStop，如果validation loss 连续100代不下降，则终止
# CheckPoint，只保存最优模型的权重（对应Validation Loss最小的情况）
# callbacks = [xxxx, xxxx, xxxx]
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min'),
    EarlyStopping(monitor='val_loss', patience=100, mode='min'),
    ModelCheckpoint(filepath=os.path.join(store_folder, 'best_weights.h5'), monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
]


model.compile(loss=dice_coef_loss, optimizer=Adam(0.001), metrics=['accuracy'])
history = model.fit(x=training_data, y=training_label, epochs=1000, batch_size=12, verbose=1,
                    validation_data=(validation_data, validation_label), callbacks=callbacks)

model.save_weights(os.path.join(store_folder, 'last_weights.h5'))
show_train_history(history, 'loss', 'val_loss')
