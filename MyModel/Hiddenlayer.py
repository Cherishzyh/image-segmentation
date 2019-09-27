from keras import backend as K
import matplotlib.pyplot as plt
import os
K.set_image_data_format('channels_last')

def SaveFeatureMap(model, one_data, feature_map_root_folder):
    '''
    对keras建立的模型，通过输入一个输入数据，将该输入在网络中通过正向传播，得到的所有feature map进行保存。
    :param model: keras 已存的模型，目前已通过基于TensorFlow的测试
    :param one_data: 4D数据，数据组成为 1 x row x col x channel
    :param feature_map_root_folder: 用以存储的文件夹名
    :return: None

    Apr-30-2019
    '''

    if not os.path.exists(feature_map_root_folder):
        os.mkdir(feature_map_root_folder)
        print('The folder is created: {}'.format(feature_map_root_folder))
    if not os.path.isdir(feature_map_root_folder):
        print('The root folder is not a folder: {}'.format(feature_map_root_folder))
        return None
    if one_data.ndim != 4:
        print('The dimension of the input_data should be 4')
        return None
    if one_data.shape[0] != 1:
        print('The first dimmension (the number of samples) should be 1')
        return None

    # 获得各个隐含层的名字
    layer_name = []
    for layer in model.layers:
        layer_name.append(layer.name)

    for layer_index, layer in enumerate(model.layers):
        # 输入层跳过
        if layer_index == 0:
            continue

        one_hidden_layer_folder = os.path.join(feature_map_root_folder, str(layer_index) + '_' + layer_name[layer_index])

        if not os.path.exists(one_hidden_layer_folder):
            os.mkdir(one_hidden_layer_folder)

        # get feature map in i+1 layer
        get_layer_output = K.function([model.layers[0].input], [model.layers[layer_index].output])
        # get_layer_output = K.function([one_data[0], one_data[1], one_data[2]], [model.layers[layer_index].output])

        # get the feature map of a picture
        output_layer_list = get_layer_output([one_data])
        output_layer = output_layer_list[0]
        # output_layer_list = get_layer_output(one_data)
        # output_layer = output_layer_list

        # 保存该隐含层中的每个channel
        for k in range(output_layer.shape[-1]):
            layer_output = output_layer[..., k]

            one_specific_layer_path = os.path.join(one_hidden_layer_folder, str(k) + '.png')

            plt.axis('off')
            plt.imsave(one_specific_layer_path, layer_output[0, :, :], format="png", cmap='gray')


def main():
    from MyModel.SaveAndLoad import ReadModel
    testing_folder = r'D:\data\Challenge\Map3_256\DivideData512\Test'
    path = r'C:/Users/I/Desktop/hidden'
    model_path = r'D:\data\Challenge\Map3_256\model\model.yaml'
    best_weights_path = r'HD:\data\Challenge\Map3_256\model\best_weights.h5'
    from CNNModel.Training.Generate import ImageInImageOut2DTest
    input_list, output_list, pred_list = ImageInImageOut2DTest(testing_folder, input_shape=(496, 496, 3))
    model = ReadModel(model_path, best_weights_path)
    SaveFeatureMap(model, pred_list, path)


if __name__ == '__main':
    main()