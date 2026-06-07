from config.configurator import configs
from data_utils.data_handler_sequential import DataHandlerSequential

def build_data_handler():
    data_type = configs['data']['type']
    
    if data_type == 'sequential':
        return DataHandlerSequential()
    else:
        raise NotImplementedError('Data type {} is not implemented'.format(data_type))
