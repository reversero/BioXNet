#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from os.path import join
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_fns, optimizer, config,
                 train_data_loader, valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None, 
                 class_weights=None, class_num=2, n_outputs=1, loss_weights=[1], 
                 prediction_output='average', save_attention=False):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader) # 最大 epoch 数 = 批次数量
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        
        self.class_weights = class_weights
        self.class_num = class_num
        self.n_outputs = n_outputs
        self.loss_weights = loss_weights
        self.prediction_output = prediction_output
        self.save_attention = save_attention

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)


    '''
        计算损失函数:
            - 如果输出有多个，即 self.n_outputs > 1，则遍历每个输出，并根据权重 self.loss_weights 计算加权损失
                损失的计算通过 self.criterion 完成，如果存在类别权重 self.class_weights，则将其应用于损失计算中
            - 如果只有一个输出，即 self.n_outputs == 1，则直接计算输出和目标之间的损失。同样，如果存在类别权重 self.class_weights，也会将其应用于损失计算中
    '''
    def _compute_loss(self, outcome, decision_outcomes, target):
        # loss = self._compute_loss(outcome=outcome, decision_outcomes=decision_outcomes, target=target) # 计算损失函数
        if self.n_outputs > 1:
            output = decision_outcomes
            loss = 0
            for i, lw in enumerate(self.loss_weights): # "loss_weights": [2, 7, 20, 54, 148, 400]。确定该超参数的依据？
                if self.class_weights is None:
                    loss += lw * self.criterion(output[i], target)
                else:
                    class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(target.device)
                    loss += lw * self.criterion(output[i], target, class_weights)
            return loss
        else:
            output = outcome
            if self.class_weights is None:
                return self.criterion(output, target)
            else:
                class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(target.device)
                return self.criterion(output, target, class_weights)


    '''
        获取预测分数:
            - 如果有多个输出（self.n_outputs > 1），则根据 self.prediction_output 参数来计算预测分数：
                - 如果 self.prediction_output 为 'average'，则计算所有输出的平均值
                - 如果 self.prediction_output 为 'weighted'，则根据损失权重 self.loss_weights 加权计算输出的加权平均值
                - 如果 self.prediction_output 其他值，则取最后一个输出
            - 如果只有一个输出（self.n_outputs == 1），则根据 self.class_num 参数和输出类型进行如下处理：
                - 如果是多类别分类（self.class_num > 2），则对输出进行 softmax 归一化处理
                - 如果是二分类（self.class_num == 2），则对输出进行 sigmoid 处理
    '''
    def _get_prediction_scores(self, outcome, decision_outcomes):
        if self.n_outputs > 1:
            if self.class_num is not None and self.class_num > 2:
                # multiclass classification
                decision_outcomes = [torch.log_softmax(v, dim=1) for v in decision_outcomes]
            if self.class_num is not None and self.class_num == 2:
                # binary classification
                decision_outcomes = [torch.sigmoid(v) for v in decision_outcomes]

            output = [a.cpu().detach().numpy() for a in decision_outcomes]
            if self.prediction_output == 'average':
                prediction_score = np.mean(output, axis=0)
            elif self.prediction_output == 'weighted':
                prediction_score = np.average(output, axis=0, weights=self.loss_weights)
            else:
                prediction_score = output[-1]
                
        else:
            if self.class_num is not None and self.class_num > 2:
                outcome = torch.log_softmax(outcome, dim=1)
            if self.class_num is not None and self.class_num == 2:
                outcome = torch.sigmoid(outcome)
            prediction_score = outcome.cpu().detach().numpy()
        return prediction_score

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (_, _, data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            if self.n_outputs > 1:
                decision_outcomes, _ = self.model(data)
                outcome = None
            else:
                outcome, _ = self.model(data)
                decision_outcomes = None

            loss = self._compute_loss(outcome=outcome, decision_outcomes=decision_outcomes, target=target) # 计算损失函数
            loss.backward() # 反向传播
            self.optimizer.step() # optimizer 更新参数

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item()) # 使用 update 方法将新的指标值添加到指定的指标名称下，在训练过程中跟踪和记录各种指标的变化
            with torch.no_grad(): # 不计算梯度，计算预测结果，并更新评估指标
                y_pred = self._get_prediction_scores(outcome, decision_outcomes).squeeze()
                y_true = target.cpu().detach().numpy().squeeze()
                for met in self.metric_fns:
                    if met.__name__ not in ['roc_auc', 'multi_precision']:
                        self.train_metrics.update(met.__name__, met(y_pred, y_true))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_'+k : v for k, v in val_log.items()}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        result_dict = {'y_true': [], 'y_pred': []}
        with torch.no_grad():
            for batch_idx, (_, _, data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if self.n_outputs > 1:
                    decision_outcomes, _ = self.model(data)
                    outcome = None
                else:
                    outcome, _ = self.model(data)
                    decision_outcomes = None

                loss = self._compute_loss(outcome=outcome, decision_outcomes=decision_outcomes, target=target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                y_pred = self._get_prediction_scores(outcome, decision_outcomes).squeeze()
                y_true = target.cpu().detach().numpy().squeeze()
                result_dict['y_pred'].append(y_pred)
                result_dict['y_true'].append(y_true)
        
        result_dict['y_pred'] = np.concatenate(result_dict['y_pred'])
        result_dict['y_true'] = np.concatenate(result_dict['y_true'])
        for met in self.metric_fns:
            self.valid_metrics.update(met.__name__, met(result_dict['y_pred'], result_dict['y_true']))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self, test_result_filename=None):
        self.model.eval()
        # 创建一个字典 result_dict，用于存储测试结果，其中包含了样本ID、药物ID、真实标签和预测标签
        result_dict = {'sample_id': [], 'drug_id': [],
                       'y_true': [], 'y_pred': []}
        # attention_list = []
        with torch.no_grad():
            for batch_idx, (sample_id, drug_id, data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if self.n_outputs > 1:
                    decision_outcomes, attention_probs_list = self.model(data)
                    outcome = None
                else:
                    outcome, attention_probs_list = self.model(data)
                    decision_outcomes = None

                y_pred = self._get_prediction_scores(outcome, decision_outcomes).squeeze()
                y_true = target.cpu().detach().numpy().squeeze()
                
                result_dict['sample_id'] += list(sample_id)
                result_dict['drug_id'] += list(drug_id)
                result_dict['y_pred'].append(y_pred)
                result_dict['y_true'].append(y_true)
                # attention_list.append([v.cpu().detach().numpy() for v in attention_probs_list])
        
        test_metrics = {}
        result_dict['y_pred'] = np.concatenate(result_dict['y_pred'])
        result_dict['y_true'] = np.concatenate(result_dict['y_true'])
        for met in self.metric_fns:
            test_metrics[met.__name__] = met(result_dict['y_pred'], result_dict['y_true'])

        if self.class_num is not None and self.class_num > 2:
            test_output = pd.DataFrame({'sample_id': result_dict['sample_id'],
                                        'drug_id': result_dict['drug_id'],
                                        'y_true': list(result_dict['y_true'].flatten()),
                                        'y_pred': list(np.argmax(result_dict['y_pred'], axis=1).flatten())})
        else:
            test_output = pd.DataFrame({'sample_id': result_dict['sample_id'],
                                        'drug_id': result_dict['drug_id'],
                                        'y_true': list(result_dict['y_true'].flatten()),
                                        'y_pred': list(result_dict['y_pred'].flatten())})
        if test_result_filename is not None:
            test_output.to_csv(join(self.config.log_dir, test_result_filename), index=False)
        else:
            test_output.to_csv(join(self.config.log_dir, 'test_output.csv'), index=False) # 结果保存到 log 文件夹中

        return test_metrics

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
