from models.conv_model import ConvModel


class ModelPicker:
    def __init__(self, config):
        self.config = config
        self.models = {
            'conv': ConvModel(config['conv'])
        }

    def pick_model(self, model=''):
        return_model = None
        if model != '' and model not in self.models:
            print('%s not match any available model')
        if model == '':
            model_name = self.config['choose_model']
        else:
            model_name = model
        return self.models[model_name]
