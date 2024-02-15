#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset

class BioXNetDataset(Dataset):
    def __init__(self, response_array,
                       mutation_dict,
                       cnv_dict,
                       methylation_dict,
                       drug_target_dict,
                       input_data_order):
        self.response_array = response_array
        self.mutation_dict = mutation_dict
        self.cnv_dict = cnv_dict
        self.methylation_dict = methylation_dict
        self.drug_target_dict = drug_target_dict    
        self.input_data_order = input_data_order

    def __len__(self):
        return len(self.response_array)


    '''
        根据索引 idx 获取数据集中的一个样本:
            - 根据索引从 response_array 中获取样本的ID、药物ID和响应值。然后，分别从四个数据字典中获取样本的突变、CNV、甲基化和药物靶标信息
            - 根据 input_data_order 参数决定数据的顺序，构建输入列表 input_list
            - 将 input_list 转换为 PyTorch 的张量 input_tensor 和 response_tensor
            - 返回样本ID、药物ID、输入张量和响应张量
    '''
    def __getitem__(self, idx):
        sample_id, drug_id, response = self.response_array[idx]

        # get the row as list in mutation_df by sample_id
        mutation = self.mutation_dict[sample_id]
        cnv = self.cnv_dict[sample_id]
        methylation = self.methylation_dict[sample_id]
        drug_target = self.drug_target_dict[drug_id]

        # input_list = self._build_input(mutation, cnv, methylation, drug_target, self.input_data_order)
        if self.input_data_order == ["drug_target", "mutation", "cnv"]:
            input_list = [val for pair in zip(drug_target, mutation, cnv) for val in pair]
        elif self.input_data_order == ["drug_target", "mutation", "methylation"]:
            input_list = [val for pair in zip(drug_target, mutation, methylation) for val in pair]
        elif self.input_data_order == ["drug_target", "cnv", "methylation"]:
            input_list = [val for pair in zip(drug_target, cnv, methylation) for val in pair]
        elif self.input_data_order == ["drug_target", "mutation"]:
            input_list = [val for pair in zip(drug_target, mutation) for val in pair]
        else:
            input_list = [val for pair in zip(drug_target, mutation, cnv, methylation) for val in pair]
        input_tensor = torch.tensor(input_list, dtype=torch.float)
        response_tensor = torch.tensor([response], dtype=torch.float)
        return sample_id, drug_id, input_tensor, response_tensor

    def _build_input(self, mutation, cnv, methylation, drug_target, input_data_order):
        # create a new list whose elements are the elements in these four lists appearing alternately
        input_list = []
        for i in range(len(mutation)):
            for data_type in input_data_order:
                if data_type == 'mutation':
                    input_list.append(mutation[i])
                elif data_type == 'cnv':
                    input_list.append(cnv[i])
                elif data_type == 'methylation':
                    input_list.append(methylation[i])
                elif data_type == 'drug_target':
                    input_list.append(drug_target[i])
        return input_list