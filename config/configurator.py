import os
import yaml
import argparse

def parse_configure():
    parser = argparse.ArgumentParser(description='CICRec')
    parser.add_argument('--model', type=str, default='cicrec', help='Model name')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args = parser.parse_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    model_name = args.model.lower()
    config_path = './config/modelconf/{}.yml'.format(model_name)
    
    if not os.path.exists(config_path):
        raise Exception("Configuration file not found: {}".format(config_path))

    with open(config_path, encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)

        # model name
        configs['model']['name'] = configs['model']['name'].lower()

        # gpu device
        configs['device'] = args.device

        # dataset
        if args.dataset is not None:
            configs['data']['name'] = args.dataset

        # log
        if 'log_loss' not in configs['train']:
            configs['train']['log_loss'] = True

        # early stop
        if 'patience' in configs['train']:
            if configs['train']['patience'] <= 0:
                raise Exception("'patience' should be greater than 0.")
            else:
                configs['train']['early_stop'] = True
        else:
            configs['train']['early_stop'] = False

        return configs

configs = parse_configure()
