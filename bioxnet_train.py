import argparse
import collections
import torch
import numpy as np
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.bioxnet as module_arch
from parse_config import ConfigParser
from trainer import Trainer


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    SEED = config['data_loader']['args']['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # setup data_loader instances
    logger.info('------------------Setup dataloader--------------------')
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    if config['trainer']['class_weights'] != None:
        get_class_weights = True
    else:
        get_class_weights = False
    train_data_loader, validate_data_loader, test_data_loader, class_weights = data_loader.get_dataloader(get_class_weights)
    reactome_network = data_loader.get_reactome()
    features, genes = data_loader.get_features_genes()
    logger.info('------------------Finish dataloader-------------------')

    # build model architecture, then print to console
    logger.info('------------------Setup model-------------------------')
    config['arch']['args']['features'] = features
    config['arch']['args']['genes'] = genes
    config['arch']['args']['reactome_network'] = reactome_network
    config['arch']['args']['logger'] = logger
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    
    if config['pretrain'] is not None:
        logger.info('Loading the pre-trained model from {} ...'.format(config['pretrain']))
        checkpoint = torch.load(config['pretrain'])
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        logger.info('Training the model from scratch ...')
    
    if config['transfer_directly'] == True:
        logger.info('Transfer the model directly, only train prediction layer.')
        for name, param in model.named_parameters():
            if 'output' not in name:
                param.requires_grad = False

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logger.info(f'Trainable parameters {params}.')
    logger.info('------------------Finish setting up model--------------')

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=validate_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      class_weights=class_weights,
                      class_num=config['arch']['args']['class_num'],
                      n_outputs=config['arch']['args']['n_outputs'],
                      loss_weights=config['trainer']['loss_weights'],
                      prediction_output=config['trainer']['prediction_output'])

    logger.info('------------------Start training-----------------------')
    trainer.train()
    logger.info('------------------Finish training---------------------')

    """Test."""
    logger.info('------------------Start testing-----------------------')
    logger = config.get_logger('test')
    logger.info(model)
    
    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    test_output = trainer.test()
    logger.info(test_output)
    logger.info('------------------Finish testing----------------------')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
