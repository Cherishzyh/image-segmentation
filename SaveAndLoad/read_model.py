from keras.models import model_from_yaml


def ReadModel(model_path, weights_path):
    # load YAML and create model
    yaml_file = open(model_path)
    loaded_model = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model)

    # load weights into new model
    model.load_weights(weights_path)

    return model




