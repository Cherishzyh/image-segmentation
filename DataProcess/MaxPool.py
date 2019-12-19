import numpy as np


# 我的四维
def pooling(feature_map, size=2, stride=2):
    height = feature_map.shape[0]
    width = feature_map.shape[1]
    channel = feature_map.shape[2]
    padding_height = np.uint16(round((height - size + 1) / stride))
    padding_width = np.uint16(round((width - size + 1) / stride))
    #
    pool_out = np.zeros((padding_height, padding_width, channel), dtype=np.uint8)

    for channel in range(feature_map.shape[2]):
        for width in range(0, feature_map.shape[1], 2):
            for height in range(0, feature_map.shape[0], 2):
                pool_out[int(height/stride), int(width/stride), channel] = \
                    np.max(feature_map[height:height+2, width:width+2, channel])

    return pool_out


# 这复制
def maxpooling(feature_map, size=2, stride=2):
    channel = feature_map.shape[2]
    height = feature_map.shape[0]
    width = feature_map.shape[1]
    padding_height = np.uint16(round((height - size + 1) / stride))
    padding_width = np.uint16(round((width - size + 1) / stride))
    # print(padding_height, padding_width)

    pool_out = np.zeros((padding_height, padding_width, channel), dtype=np.uint8)

    for map_num in range(channel):
        out_height = 0
        for r in np.arange(0, height, stride):
            out_width = 0
            for c in np.arange(0, width, stride):
                pool_out[out_height, out_width, map_num] = np.max(feature_map[r:r + size, c:c + size, map_num])
                out_width = out_width + 1
            out_height = out_height + 1
    # print(pool_out.shape)
    return pool_out


# 这是师姐的代码
def MaxPool(data):
    s0 = np.shape(data)[0]
    s1 = np.shape(data)[1]
    s2 = np.shape(data)[2]
    sampled_data = np.zeros([s0, int(s1/2), int(s2 / 2), 1])
    for z in range(s0):
        im = data[z, :, :, 0]
        resized_im = np.zeros([int(s1/2), int(s2 / 2)])
        for i in range(int(s1/2)):
             for j in range(int(s2 / 2)):
                area = im[2*i: 2*i + 2, 2*j: 2*j + 2]
                pixel = np.max(area[:])
                resized_im[i, j] = pixel
        sampled_data[z, :, :, 0] = resized_im
    return sampled_data


def main():
    import h5py
    import matplotlib.pyplot as plt
    data_folder = r'D:\ZYH\Data\TZ roi\TypeOfData\FormatH5\Problem'
    from DataProcess.Data import GetData
    image, label, _ = GetData(data_folder)
    # data_path = r'D:\ZYH\Data\TZ roi\TypeOfData\FormatH5\Problem\Chen ren geng-slicer_index_10.h5'
    # with h5py.File(data_path, 'r') as h5_file:
    #     # image = np.asarray(h5_file['input_0'], dtype=np.float32)
    #     label = np.asarray(h5_file['output_0'], dtype=np.uint8)

    for i in range(len(label)):
        pool_label = max_pool(label)
        plt.subplot(121)
        plt.title('pool')
        plt.imshow(pool_label[i, ..., 0], cmap='gray')
        plt.subplot(122)
        plt.title('original')
        plt.imshow(label[i, ..., 1], cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()