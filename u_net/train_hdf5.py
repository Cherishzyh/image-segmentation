from u_net import u_net
from data import GetData
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from filepath import train_data_folder, validation_data_folder, store_folder
import os


training_data, training_label = GetData(train_data_folder)
validation_data, validation_label = GetData(validation_data_folder)

# training_data.shape = (number, row, col, channel)
model = u_net(training_data.shape[1:])
print(model.summary())

'''Save model'''
model_yaml = model.to_yaml()
with open(os.path.join(store_folder, 'model.yaml'), "w") as yaml_file:
    yaml_file.write(model_yaml)

#callbacks
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

from keras.optimizers import Adam
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
history = model.fit(x=training_data, y=training_label, epochs=1, batch_size=1, verbose=1,
                    validation_data=(validation_data, validation_label), callbacks=callbacks)

model.save_weights(os.path.join(store_folder, 'last_weights.h5'))
