import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

'''
    基础的训练器类，用于在训练神经网络模型时进行通用操作:
        - 将模型、损失函数、优化器等移动到指定的设备上，并支持多 GPU 训练
        - 定义了训练一个 epoch 的逻辑，具体的训练过程需要在子类中实现
        - 实现了整个训练流程，包括保存模型、监控指标、提前停止等功能
        - 支持从断点恢复训练
'''
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_fns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_fns = metric_fns
        self.optimizer = optimizer
        self.config = config

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            # mnt_mode determines minimum or maximum metrix is best, if we use loss, thus mnt_mode = min
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError # 报错


    '''
        执行完整的模型训练:
            - 迭代每个 epoch，调用 _train_epoch 方法执行训练，并将结果记录到日志中
            - 评估模型性能，并根据指定的监控指标决定是否保存模型的检查点
            - 如果设置了保存最佳模型，保存最佳模型的检查点
            - 如果模型性能在一定数量的 epoch 中没有改善，则停止训练
    '''
    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            self.logger.info('epoch: {}'.format(epoch))
            for key in ['train', 'validation']:
                if key not in log:
                    continue
                value_format = ''.join(['{:15s}: {:.2f}\t'.format(k, v) for k, v in log[key].items()])
                self.logger.info('    {:15s}: {}'.format(str(key), value_format))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    # improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                    #            (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    # use minus to make sure the loss is decreasing
                    improved = (self.mnt_mode == 'min' and (log[self.mnt_metric] - self.mnt_best) <= 1e-4) or \
                               (self.mnt_mode == 'max' and (log[self.mnt_metric] - self.mnt_best) >= 1e-4)
                    '''
                    在尝试检查性能改善时，会根据设定的监视模式（最小化或最大化）和指定的指标 mnt_metric，来判断性能是否有所提升
                    具体来说，如果监视模式是最小化，那么只有当当前指标值比历史最佳指标值小一定的阈值（这里是1e-4）时，才认为性能有所提升
                    如果监视模式是最大化，那么只有当当前指标值比历史最佳指标值大一定的阈值时才认为性能有所提升
                    这里的1e-4是一个可调的参数，可以根据具体情况进行调整
                    （有的指标值越大，模型性能越好，e.g., R2，有的则越小越好，e.g., MSE）
                    '''
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self._save_checkpoint(epoch, save_best=best)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)


    '''
        准备设备:
            - 通过 torch.cuda.device_count() 获取可用的 GPU 数量
            - 检查用户配置的 GPU 数量是否大于 0，如果是且没有可用的 GPU，则发出警告并将 n_gpu_use 设置为 0，表示将在 CPU 上进行训练
            - 如果用户配置的 GPU 数量大于可用的 GPU 数量，则发出警告，并将 n_gpu_use 设置为可用的 GPU 数量
            - 根据 n_gpu_use 的值选择设备，如果大于 0，则选择第一个 GPU，否则选择 CPU
    '''
    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids


    '''
        保存训练过程中的检查点:
            - 获取模型的名称，并将其存储在变量 arch 中
            - 创建一个字典 state，其中包含了要保存的信息，包括模型的状态字典、优化器的状态字典、当前的 epoch 数、最佳监控指标值、以及配置信息
            - 如果 save_best 参数为 True，则将检查点保存为 model_best.pth，如果存在属性 current_k，则在文件名中包含当前 K 值
            - 如果 save_best 参数为 False，则将检查点保存为 checkpoint-epoch{}.pth，其中 {} 会被当前的 epoch 数替换
    '''
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            if hasattr(self, 'current_k'):
                best_path = str(self.checkpoint_dir / 'model_best_K{}.pth'.format(self.current_k))
            else:
                best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))


    '''
        从已保存的检查点中恢复训练:
            - 将输入的检查点路径转换为字符串类型，并记录日志，指示正在加载检查点
            - 使用 torch.load 加载检查点文件，将其保存在 checkpoint 变量中
            - 获取检查点中的 epoch 数，并将其加1，以便从下一个 epoch 开始训练
            - 更新监视的最佳指标值，以便在需要时更新模型的性能
            - 检查配置文件中的模型架构是否与检查点中的架构匹配，如果不匹配，则记录警告。然后加载模型的状态字典
            - 检查配置文件中的优化器类型是否与检查点中的类型匹配，如果不匹配，则记录警告。然后加载优化器的状态字典
            - 记录日志，指示检查点已加载完成，并显示从哪个 epoch 开始恢复训练
    '''
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
