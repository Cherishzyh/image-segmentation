from keras.models import model_from_yaml
from data import GetData



yaml_file = open(r'H:/data/SaveModel/model.yaml')
loaded_model = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model)
