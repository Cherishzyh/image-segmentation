from data import GetData
import matplotlib.pyplot as plt
from read_model import ReadModel
from filepath import test_data_folder, model_path, last_weights_path
from dice import dice_coef_loss

# load test data
image_test, label_test = GetData(test_data_folder)

# load model
model = ReadModel(model_path, last_weights_path)
model.compile(loss=dice_coef_loss, optimizer='adam', metrics=['accuracy'])

# evaluate loaded model on test data
# score = model.evaluate(x=image_test, y=label_test)
# print('accuracy = ', score[1])

# prediction loaded model on test data
prediction = model.predict(image_test)

# for i in range(len(prediction)):
#     plt.contour(label_test[i, :, :, 0], colors='r')
#     plt.contour(prediction[i, :, :, 0], colors='g')
#     plt.imshow(image_test[i, :, :, 0], cmap='gray')
#     plt.show()
