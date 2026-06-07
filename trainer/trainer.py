import random
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.bulid_model import build_model
from copy import deepcopy


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=optim_config['lr'], 
                weight_decay=optim_config['weight_decay']
            )

    def train_epoch(self, model, epoch_idx):
        # Prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # For recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        
        # Start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # Record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # Log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

    def train(self, model):
        self.create_optimizer(model)
        train_config = configs['train']

        if not train_config['early_stop']:
            for epoch_idx in range(train_config['epoch']):
                # Train
                self.train_epoch(model, epoch_idx)
                # Evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx)
            self.test(model)
            return model

        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            
            for epoch_idx in range(train_config['epoch']):
                # Train
                self.train_epoch(model, epoch_idx)
                # Evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][0]][0]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased. Copying the best model...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    # Early stop
                    if now_patience == configs['train']['patience']:
                        break

            # Re-initialize the model and load the best parameter
            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            return model

    def evaluate(self, model, epoch_idx=None):
        model.eval()
        
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx)
        else:
            raise NotImplementedError
        
        return eval_result

    def test(self, model):
        model.eval()
        
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        else:
            raise NotImplementedError
        
        return eval_result
