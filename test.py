from ImageSegmentation.data import GetData
from ImageSegmentation.saveandload import ReadModel
from filepath import test_data_folder, model_path, best_weights_path

# load test data
image_test, label_test = GetData(test_data_folder)

# load model
model = ReadModel(model_path, best_weights_path)





