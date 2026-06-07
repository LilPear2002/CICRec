from trainer.trainer import Trainer

def build_trainer(data_handler, logger):
    return Trainer(data_handler, logger)
