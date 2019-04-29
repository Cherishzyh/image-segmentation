from keras import backend as K
from data import GetData
from saveandload import ReadModel
# from filepath import test_data_folder, model_path, best_weights_path
import matplotlib.pyplot as plt
import os
import numpy as np

K.set_image_data_format('channels_last')

test_data_folder = r'v:\ProstateSegment\data\Input_1_Ouput_1\testing'
image_test, label_test = GetData(test_data_folder)
image_test = image_test[[0], ...]
print(image_test.shape)

model_path = r'C:\Users\yangs\Desktop\temp_zyh\model.yaml'
best_weights_path = r'C:\Users\yangs\Desktop\temp_zyh\best_weights.h5'
model = ReadModel(model_path, best_weights_path)

layer_name = []
for layer in model.layers:
    layer_name.append(layer.name)

featur_map_root_folder = r'C:\Users\yangs\Desktop\temp_zyh\feature_map'
for i, layer in enumerate(model.layers):
    if i == 0:
        continue

    one_hidder_layer_folder = os.path.join(featur_map_root_folder, layer_name[i])

    if not os.path.exists(one_hidder_layer_folder):
        os.mkdir(one_hidder_layer_folder)

    # get feature map in i+1 layer
    get_layer_output = K.function([model.layers[0].input], [model.layers[i].output])

    # get the feature map of a picture
    output_layer_list = get_layer_output([image_test])
    print(len(output_layer_list))
    output_layer = output_layer_list[0]

    for k in range(output_layer.shape[-1]):
        layer_output = output_layer[..., k]

        one_specific_layer_path = os.path.join(one_hidder_layer_folder, str(k) + '.png')

        plt.axis('off')
        plt.imsave(one_specific_layer_path, layer_output[0, :, :], format="png", cmap='gray')


