from keras import backend, Model
from data import GetData
from saveandload import ReadModel
from filepath import test_data_folder, model_path, best_weights_path
import matplotlib.pyplot as plt
import os

image_test, label_test = GetData(test_data_folder)

model = ReadModel(model_path, best_weights_path)

layer_name = []
for layer in model.layers:
    layer_name.append(layer.name)

# for i in range(len(model.layers)):
#     hidden_layer = model.get_layer(index=i)
#     new_model = Model(inputs=model.input, outputs=hidden_layer.output)
#     result = new_model.predict(image_test)
#     print(result.shape)
#     plt.imshow(result[5, :, :, 0], cmap='gray')

for i in range(len(model.layers)):
    picture_path = r'H:/data/SaveModel/model/picture'

    picture_path = os.path.join(picture_path, layer_name[i+1])
    os.mkdir(picture_path)

    # get feature map in i+1 layer
    get_layer_output = backend.function([model.layers[0].input], [model.layers[i+1].output])

    # get the feature map of a picture
    for k in range(len(get_layer_output([image_test]))):

        layer_output = get_layer_output([image_test])[k]

        feature_map_name = str(k)
        picture_path = os.path.join(picture_path, feature_map_name)

        plt.axis('off')
        plt.imsave(picture_path, layer_output[0, :, :, 0], format="png", cmap='gray')


