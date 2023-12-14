#!/usr/bin/env python3
import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn

class Diagonal(nn.Module):
    def __init__(self, output_dim, use_bias=True, input_shape=None):
        super(Diagonal, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        input_dim = input_shape[1]
        self.n_inputs_per_node = input_dim // self.output_dim

        # create parameter
        self.kernel = nn.Parameter(torch.randn(1, input_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
        self.__initializer()

    def __initializer(self):
        torch.nn.init.xavier_uniform_(self.kernel)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # kernel is used to assign a weight to each input, shape: [1, genes*4]
        mult = x * self.kernel # [batch_size, genes*4], element-wise multiplication
        mult = torch.reshape(mult, (-1, self.n_inputs_per_node)) # [batch_size*genes, 4]
        # for each gene, sum the weighted four input together as the output
        mult = torch.sum(mult, dim=1) # [batch_size*genes]
        output = torch.reshape(mult, (-1, self.output_dim)) # [batch_size, genes]

        if self.use_bias:
            output = output + self.bias

        return output

class SparseTF(nn.Module):
    def __init__(self, output_dim, map=None, use_bias=True, input_shape=None):
        '''Sparse Tensor Factorization layer
        
        Args:
            output_dim (_type_): _description_
            map (np.array, binary): the shape is (input_dim, output_dim) and the value is 1 or 0
            use_bias (bool, optional): _description_. Defaults to True.
            input_shape (_type_, optional): _description_. Defaults to None.
        '''
        super(SparseTF, self).__init__()
        self.output_dim = output_dim
        input_dim = input_shape[1]
        self.use_bias = use_bias
        
        self.register_buffer('map', map)

        self.kernel = nn.Parameter(torch.randn(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
        self.__initializer()

    def __initializer(self):
        torch.nn.init.xavier_uniform_(self.kernel)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs):
        # shape of inputs: (batch_size, input_dim)
        with torch.no_grad():
            self.kernel.mul_(self.map)
        # map = self.map.to(inputs.device)
        # tt = self.kernel * map # element-wise multiplication, shape (input_dim, output_dim)
        # output = torch.matmul(inputs, tt)
        output = torch.matmul(inputs, self.kernel)
        if self.use_bias:
            output = output + self.bias
        return output


class Dense(nn.Module):
    def __init__(self, output_dim, input_shape=None, use_bias=True):
        super(Dense, self).__init__()
        input_dim = input_shape[1]
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.kernel = nn.Parameter(torch.randn(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
        self.__initializer()

    def __initializer(self):
        torch.nn.init.xavier_uniform_(self.kernel)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = torch.mm(x, self.kernel) # shape: (batch_size, output_dim)
        if self.use_bias:
            x = x + self.bias
        return x

class BioXNetLayer(nn.Module):
    def __init__(self, mapp, dropout, sparse, use_bias, attention, batch_normal, class_num, i):
        super(BioXNetLayer, self).__init__()
        self.attention = attention
        self.batch_normal = batch_normal
        self.class_num = class_num
        self.i = i

        n_genes, n_pathways = mapp.shape
        if sparse:
            mapp = self.df_to_tensor(mapp)
            self.hidden_layer = SparseTF(n_pathways, mapp, use_bias=use_bias, input_shape=(None, n_genes))
        else:
            self.hidden_layer = Dense(n_pathways, input_shape=(None, n_genes))
        self.activation = nn.Tanh()
        
        if attention:
            self.attention_prob_layer = Dense(n_pathways, input_shape=(None, n_genes))
            self.attention_activation = nn.Sigmoid()
        
        # testing
        if self.class_num == 2 or self.class_num is None:
            # binary classification or regression, the output is a single number
            self.decision_layer_single_output = nn.Linear(in_features=n_pathways, out_features=1)
            # if self.class_num == 2:
            #     self.decision_activation = nn.Sigmoid()
        else:
            # multi-class classification, the output is a vector
            self.decision_layer_multi_output = nn.Linear(in_features=n_pathways, out_features=self.class_num)
        if batch_normal:
            self.batchnormal_layer = nn.BatchNorm1d(num_features=n_pathways)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs):
        outcome = self.hidden_layer(inputs)
        if self.batch_normal:
            outcome = self.batchnormal_layer(outcome)
        outcome = self.activation(outcome)
        if self.attention:
            attention_probs = self.attention_activation(self.attention_prob_layer(inputs))
            outcome = outcome * attention_probs
        else:
            attention_probs = None
        outcome = self.dropout_layer(outcome)

        if self.class_num == 2 or self.class_num is None:
            decision_outcome = self.decision_layer_single_output(outcome)
        else:
            decision_outcome = self.decision_layer_multi_output(outcome)

        # if self.class_num == 2:
        #     decision_outcome = self.decision_activation(decision_outcome)
    
        return outcome, decision_outcome, attention_probs

    def df_to_tensor(self, df):
        tensor = torch.from_numpy(df.to_numpy())
        tensor = tensor.type(torch.FloatTensor)
        return tensor


class BioXNet(nn.Module):
    def __init__(self, 
                 features, 
                 genes, 
                 n_hidden_layers, 
                 direction, 
                 dropout_list, 
                 sparse, 
                 add_unk_genes, 
                 batch_normal, 
                 reactome_network, 
                 class_num=2,
                 n_outputs=1,
                 use_bias=False, 
                 shuffle_genes=False, 
                 attention=False, 
                 sparse_first_layer=True, 
                 logger=None):
        '''The architecture of PNet.

        Args:
            features (df multiindex): the columns in TCGA, the first level is genes, and the second level is the data type.
            genes (df columns): the genes in TCGA data.
            n_hidden_layers (int): the number of hidden layers in Reactome
            direction (string): "root_to_leaf"
            dropout_list (list): the dropout rate for each layer
            sparse (bool): whether to use sparse architecture
            add_unk_genes (bool): _description_
            batch_normal (_type_): _description_
            reactome_network (class): the reactome network
            class_num (int, optional): 2 - binary classification; None - regression; >2 - multi-class classification. Defaults to 2.
            n_outputs (int, optional): the number of outputs. Defaults to 1.
            use_bias (bool, optional): _description_. Defaults to False.
            shuffle_genes (bool, optional): _description_. Defaults to False.
            attention (bool, optional): _description_. Defaults to False.
            sparse_first_layer (bool, optional): _description_. Defaults to True.
            logger (_type_, optional): _description_. Defaults to None.
        '''
        super(BioXNet, self).__init__()
        self.feature_names = {}
        n_features = len(features)
        n_genes = len(genes)
        self.reactome_network = reactome_network
        self.attention = attention 
        self.batch_normal = batch_normal
        self.n_hidden_layers = n_hidden_layers
        self.logger = logger
        self.class_num = class_num
        self.n_outputs = n_outputs

        '''layer1 is used to assign learning weights for each input
                shape: (n_features, n_genes)
                input: (batch_size, n_features)
                output: (batch_size, n_genes)
        '''
        if sparse:
            # the difference between SparseTF and Diagonal is that the kernel in SparseTF is in shape (n_features, n_genes)
            # while the kernel in Diagonal is in shape (1, n_features)
            if shuffle_genes == 'all':
                # create mapp randomly to shuffle the genes
                ones_ratio = float(n_features) / np.prod([n_genes, n_features])
                self.logger.info('ones_ratio random {}'.format(ones_ratio))
                # shape of mapp: (n_features, n_genes)
                mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
                self.layer1 = SparseTF(output_dim=n_genes, map=mapp, use_bias=use_bias, input_shape=(None, n_features))
            else:
                self.layer1 = Diagonal(output_dim=n_genes, input_shape=(None, n_features), use_bias=use_bias)
        else:
            if sparse_first_layer:
                self.layer1 = Diagonal(output_dim=n_genes, input_shape=(None, n_features), use_bias=use_bias)
            else:
                self.layer1 = Dense(output_dim=n_genes, input_shape=(None, n_features), use_bias=use_bias)
        
        self.layer1_activation = nn.Tanh()
        if attention:
            self.attention_prob_layer = Diagonal(n_genes, input_shape=(None, n_features))
            self.attention_activation = nn.Sigmoid() 
        self.dropout1 = nn.Dropout(dropout_list[0])

        # testing
        if self.class_num == 2 or self.class_num is None:
            # binary classification or regression, the output is a single number
            self.decision_layer1_single_output = nn.Linear(in_features=n_genes, out_features=1)
            # if self.class_num == 2:
            #     self.decision_activation1 = nn.Sigmoid()
        else:
            # multi-class classification, the output is a vector
            self.decision_layer1_multi_output = nn.Linear(in_features=n_genes, out_features=self.class_num)
        self.batchnorm_layer1 = nn.BatchNorm1d(num_features=n_genes)

        module_list = nn.ModuleList()
        if n_hidden_layers > 0:            
            maps = self.get_layer_maps(genes=genes, n_levels=n_hidden_layers, 
                                       direction=direction, add_unk_genes=add_unk_genes)
            self.logger.info(f'original dropout list {dropout_list}')
            dropouts = dropout_list[1:]
            # no considering the last map (layer)
            for i, mapp in enumerate(maps[0: -1]):
                # mapp (pd.DataFrame): the rows are genes and the columns are pathways
                dropout = dropouts[i]
                names = mapp.index
                if shuffle_genes in ['all', 'pathways']:
                    mapp = self.shuffle_genes_map(mapp)
                n_genes, n_pathways = mapp.shape
                self.logger.info('HiddenLayer-{}: n_genes {}, n_pathways {}'.format(i, n_genes, n_pathways))
                self.logger.info('layer {}, dropout {}'.format(i, dropout))
                pnet_layer = BioXNetLayer(mapp=mapp, 
                                          dropout=dropout, 
                                          sparse=sparse, 
                                          use_bias=use_bias, 
                                          attention=attention, 
                                          batch_normal=batch_normal,
                                          class_num=self.class_num,
                                          i=i)
                module_list.add_module('HiddenLayer{}'.format(i), pnet_layer)
                self.feature_names['h{}'.format(i)] = names
            i = len(maps)
            self.feature_names['h{}'.format(i-1)] = maps[-1].index
        self.module_list = module_list

        # output layer
        if self.class_num == 2 or self.class_num is None:
            # binary classification or regression, the output is a single number
            self.output_layer_single_output = nn.Sequential(
                nn.Linear(in_features=n_pathways, out_features=n_pathways),
                nn.LeakyReLU(),
                nn.Linear(in_features=n_pathways, out_features=1)
            )
        else:
            # multi-class classification, the output is a vector
            self.output_layer_multi_output = nn.Sequential(
                nn.Linear(in_features=n_pathways, out_features=n_pathways),
                nn.LeakyReLU(),
                nn.Linear(in_features=n_pathways, out_features=self.class_num)
            )
        
    def forward(self, inputs):
        decision_outcomes = []
        attention_probs_list = []
        
        # inputs: [samples, features]
        outcome = self.layer1(inputs) # [samples, genes]
        if self.batch_normal:
            outcome = self.batchnorm_layer1(outcome)
        outcome = self.layer1_activation(outcome)
        if self.attention:
            attention_probs = self.attention_activation(self.attention_prob_layer(inputs))
            outcome = outcome * attention_probs
            attention_probs_list.append(attention_probs)
        else:
            attention_probs_list.append(None)
        outcome = self.dropout1(outcome)

        # first layer
        if self.class_num == 2 or self.class_num is None:
            # binary classification or regression, the output is a single number
            decision_outcome = self.decision_layer1_single_output(outcome)
        else:
            # multi-class classification, the output is a vector
            decision_outcome = self.decision_layer1_multi_output(outcome)
        # if self.class_num == 2:
        #     decision_outcome = self.decision_activation1(decision_outcome)
        decision_outcomes.append(decision_outcome)

        if self.n_hidden_layers > 0:
            for layer in self.module_list:
                outcome, decision_outcome, attention_probs = layer(outcome)
                decision_outcomes.append(decision_outcome)
                attention_probs_list.append(attention_probs)

        # last layer
        if self.class_num == 2 or self.class_num is None:
            # binary classification or regression, the output is a single number
            outcome = self.output_layer_single_output(outcome)
        else:
            # multi-class classification, the output is a vector
            outcome = self.output_layer_multi_output(outcome)
        # if self.class_num == 2:
        #     outcome = self.output_activation(outcome)

        if self.n_outputs == 1:
            return outcome, attention_probs_list
        else:
            return decision_outcomes, attention_probs_list

    def get_layer_maps(self, genes, n_levels, direction, add_unk_genes):
        # reactome_layers is a list and each element is a dataframe
        # the order of elements in reactome_layers is from the source to the terminal nodes
        reactome_layers = self.reactome_network.get_layers(n_levels, direction) # list
        filtering_index = genes
        maps = []
        # reactomen_layers[::-1] is the reverse order
        for i, layer in enumerate(reactome_layers[::-1]):
            self.logger.info(f'layer {i}')
            # mapp (pd.DataFrame): the columns are pathways and the rows are genes
            mapp = self.get_map_from_layer(layer)
            # print the position of the nonzero elements in the first row
            # print(mapp.iloc[0, :].to_numpy().nonzero())
            filter_df = pd.DataFrame(index=filtering_index)
            self.logger.info(f'filtered_map {filter_df.shape}')
            # only keep the genes that existed in the input data
            filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
            
            self.logger.info(f'filtered_map {filter_df.shape}')

            # UNK, add a node for genes without known reactome annotation
            if add_unk_genes:
                self.logger.info('Add UNK')
                filtered_map['UNK'] = 0
                ind = filtered_map.sum(axis=1) == 0
                filtered_map.loc[ind, 'UNK'] = 1

            filtered_map = filtered_map.fillna(0)
            self.logger.info(f'filtered_map {filter_df.shape}')
            # filtering_index = list(filtered_map.columns)
            filtering_index = filtered_map.columns
            self.logger.info('layer {}, # of edges {}'.format(i, filtered_map.sum().sum()))
            maps.append(filtered_map)
        return maps

    def get_map_from_layer(self, layer_dict):
        pathways = list(layer_dict.keys())
        print('pathways', len(pathways))
        genes = list(itertools.chain.from_iterable(layer_dict.values()))
        genes = list(np.unique(genes))
        print('genes', len(genes))
        pathways.sort()
        genes.sort()

        n_pathways = len(pathways)
        n_genes = len(genes)

        mat = np.zeros((n_pathways, n_genes))
        for p, gs in layer_dict.items():
            g_inds = [genes.index(g) for g in gs]
            p_ind = pathways.index(p)
            mat[p_ind, g_inds] = 1

        df = pd.DataFrame(mat, index=pathways, columns=genes)

        return df.T

    def shuffle_genes_map(self, mapp):
        self.logger.info('shuffling')
        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
        self.logger.info('ones_ratio {}'.format(ones_ratio))
        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
        self.logger.info('random map ones_ratio {}'.format(ones_ratio))
        return mapp

    def df_to_tensor(self, df):
        tensor = torch.from_numpy(df.to_numpy())
        tensor = tensor.type(torch.FloatTensor)
        return tensor