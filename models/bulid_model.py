from config.configurator import configs
from models.cicrec import CICRec

def build_model(data_handler):
    model_name = configs['model']['name']

    model_map = {
        'cicrec': CICRec
    }

    if model_name in model_map:
        return model_map[model_name](data_handler)
    raise NotImplementedError('Model {} is not implemented'.format(model_name))
