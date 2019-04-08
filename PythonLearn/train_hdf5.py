from u_net import u_net
from data import GetData

data_folder = r'H:/data/Input_1_Output_1/training'
save_path = 'H:/data/SaveModel/training.h5'

image_list, label_list = GetData(data_folder)

model = u_net((180, 180, 1))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

try:
    model.load_weights(save_path)
    print("加载模型成功！继续训练模型")
except:
    print("加载模型失败！开始训练一个新模型")

history = model.fit(x=image_list, y=label_list, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

model.save_weights(save_path)