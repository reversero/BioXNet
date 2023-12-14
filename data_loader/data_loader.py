#!/usr/bin/env python3
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
from cellline_data_reader import CellLineBaseData
from TCGA.tcga_reader import TCGAData
from bioxnet_dataset import BioXNetDataset
from pre import get_processor
from Reactome.reactome import ReactomeNetwork

class BioXNetDataLoader():
    def __init__(self, data_source_list, 
                       params, 
                       input_data_order,
                       batch_size, 
                       seed, 
                       shuffle, 
                       num_workers, 
                       logger,
                       eval_dataset=True, 
                       pre_params=None,
                       gene_intersection=True):
        self.data_source_list = data_source_list
        self.params = params
        self.data_dir = Path(self.params['data_dir'])
        self.input_data_order = input_data_order
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.logger = logger
        self.eval_dataset = eval_dataset
        self.pre_params = pre_params
        self.gene_intersection = gene_intersection

        params['logger'] = self.logger
        params['data_list'] = self.data_source_list
        if 'TCGA' in self.data_source_list:
            self.logger.info('Loading TCGA data')
            self.data_reader = TCGAData(**params)
        elif 'GDSC' in self.data_source_list:
            self.logger.info(f'Loading {self.data_source_list} data')
            self.data_reader = CellLineBaseData(**params)
        else:
            self.logger.error('Unsupported data type')
            raise ValueError('Unsupported data type')

    def get_reactome(self):
        return ReactomeNetwork(data_dir=self.data_dir.joinpath('reactome'))

    def get_train_validate_test(self):
        return self.data_reader.train_valid_test_split(self.seed)

    def get_train_test(self):
        respone_train_array, respone_valid_array, respone_test_array = self.data_reader.train_valid_test_split(self.seed)
        # combine training and validation datasets
        respone_train_array = respone_train_array + respone_valid_array
        return respone_train_array, respone_test_array

    def get_data(self):
        self.logger.info('Transforming the dataframe to dictionary')
        cnv_allgenes_df, methylation_allgenes_df, \
            mut_allgenes_df, drug_target_allgenes_df = self.data_reader.get_features()
        # check whether the columns (genes) of mutation_df, cnv_df, methylation_df and drug_target_df are the same considering the order
        assert list(mut_allgenes_df.columns) == list(cnv_allgenes_df.columns) \
            == list(methylation_allgenes_df.columns) == list(drug_target_allgenes_df.columns)

        if self.gene_intersection:
            self.logger.info('Keep the genes that existed in GDSC and TCGA.')
            gene_gdsc_df = pd.read_csv(self.data_dir.joinpath('GDSC', 'processed', 'genes.csv'))
            gene_tcga_df = pd.read_csv(self.data_dir.joinpath('TCGA', 'processed', 'genes.csv'))
            gene_inter = list(set(gene_gdsc_df['genes']) & set(gene_tcga_df['genes']))

            gene_inter.sort()
            self.genes = gene_inter
            self.logger.info('The number of genes that existed in all datasets: {}.'.format(len(gene_inter)))
        else:
            self.logger.info('Use all genes.')
            gene_inter = list(set(mut_allgenes_df.columns) & set(cnv_allgenes_df.columns) \
                & set(methylation_allgenes_df.columns) & set(drug_target_allgenes_df.columns))
            gene_inter.sort()
            self.genes = gene_inter
            self.logger.info('The number of genes that existed in current datasets: {}.'.format(len(gene_inter)))
        
        cnv_allgenes_df = cnv_allgenes_df[gene_inter]
        methylation_allgenes_df = methylation_allgenes_df[gene_inter]
        mut_allgenes_df = mut_allgenes_df[gene_inter]
        drug_target_allgenes_df = drug_target_allgenes_df[gene_inter]

        self.cnv_allgenes_dict = cnv_allgenes_df.T.to_dict('list')
        self.methylation_allgenes_dict = methylation_allgenes_df.T.to_dict('list')
        self.mut_allgenes_dict = mut_allgenes_df.T.to_dict('list')
        self.drug_target_allgenes_dict = drug_target_allgenes_df.T.to_dict('list')

    def get_features_genes(self):
        '''Get the genes and data type in the input data.
        Returns:
           features (df multiindex): the first level is gene, second level is data type
           genes (list): the genes
        '''
        if not hasattr(self, 'genes'):
            self.get_data()
        genes = self.genes
        # create multiindex from two lists: genes and self.input_data_order
        gene_index, data_type_index = [], []
        for gene in genes:
            gene_index += [gene] * len(self.input_data_order)
            data_type_index += self.input_data_order
        features = pd.MultiIndex.from_arrays([gene_index, data_type_index], names=['gene', 'data_type'])

        return features, genes

    def get_dataloader(self, get_class_weights=False):
        self.get_data()
        response_train_array, response_valid_array, response_test_array = self.get_train_validate_test()
        if not self.eval_dataset:
            response_train_array = response_train_array + response_valid_array
        
        self.response_train_array = response_train_array
        self.response_test_array = response_test_array

        if self.pre_params:
            self.logger.info('Preprocessing the data')
        
        if self.eval_dataset:
            self.logger.info('Training size {}'.format(len(response_train_array)))
            self.logger.info('Validation size {}'.format(len(response_valid_array)))
            self.logger.info('Testing size {}'.format(len(response_test_array)))
        else:
            self.logger.info('Training size {}'.format(len(response_train_array)))
            self.logger.info('Validation size {}'.format(0))
            self.logger.info('Testing size {}'.format(len(response_test_array)))

        train_dataloader = self._create_dataloader(response_array=response_train_array)
        validate_dataloader = self._create_dataloader(response_array=response_valid_array)
        test_dataloader = self._create_dataloader(response_array=response_test_array)

        if get_class_weights:
            self.logger.info('Getting the class weights')
            class_weights = self.data_reader.get_class_weights(response_train_array=response_train_array)
        else:
            self.logger.info('Not using the class weights')
            class_weights = None

        return train_dataloader, validate_dataloader, test_dataloader, class_weights
    
    def get_dataloader_K_fold(self, current_k, K, get_class_weights=False):
        if not hasattr(self, 'mut_allgenes_dict'):
            self.get_data()
        Kfold_seed_path, filename_prefix = self.data_reader.train_valid_test_split_Kfold(K, self.seed)
        response_train_df = pd.read_csv(Kfold_seed_path.joinpath(filename_prefix + 'training_K{}.csv'.format(current_k)))
        response_test_df = pd.read_csv(Kfold_seed_path.joinpath(filename_prefix + 'test_K{}.csv'.format(current_k)))
        train_df, valid_df = train_test_split(response_train_df, test_size=0.2, random_state=self.seed)
        
        response_train_array = train_df.values.tolist()
        response_valid_array = valid_df.values.tolist()
        response_test_array = response_test_df.values.tolist()

        train_dataloader = self._create_dataloader(response_array=response_train_array)
        validate_dataloader = self._create_dataloader(response_array=response_valid_array)
        test_dataloader = self._create_dataloader(response_array=response_test_array)

        if get_class_weights:
            self.logger.info('Getting the class weights')
            class_weights = self.data_reader.get_class_weights(response_train_array=response_train_array)
        else:
            self.logger.info('Not using the class weights')
            class_weights = None

        return train_dataloader, validate_dataloader, test_dataloader, class_weights

    def get_dataset_tune(self):
        # check whether self.mut_allgenes_dict exist
        if not hasattr(self, 'mut_allgenes_dict'):
            self.get_data()
            response_train_array, response_valid_array, response_test_array = self.get_train_validate_test()
            response_train_array = response_train_array + response_valid_array
            self.response_train_array = response_train_array
            self.response_test_array = response_test_array

        train_dataset = self._create_dataset(response_train_array)
        test_dataset = self._create_dataset(response_test_array)

        return train_dataset, test_dataset

    def _create_dataset(self, response_array):
        dataset = BioXNetDataset(response_array=response_array,
                                 mutation_dict=self.mut_allgenes_dict,
                                 cnv_dict=self.cnv_allgenes_dict,
                                 methylation_dict=self.methylation_allgenes_dict,
                                 drug_target_dict=self.drug_target_allgenes_dict,
                                 input_data_order=self.input_data_order)
        return dataset

    def _create_dataloader(self, response_array):
        dataset = self._create_dataset(response_array)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle, num_workers=self.num_workers)
        return dataloader
    
    def preprocess(self, respone_train_array, respone_valid_array, respone_test_array):
        self.logger.info('preprocessing....')
        def get_feature_array(response_array):
            sample_id_list = list(response_array[:, 0])
            sample_id_df = pd.DataFrame(sample_id_list, columns=['sample_id'])
            sample_id_df.set_index('sample_id', inplace=True)
            cnv_joined_df = self.cnv_all_genes_df.join(sample_id_df, how='inner')
            methylation_joined_df = self.methylation_all_genes_df.join(sample_id_df, how='inner')
            mut_joined_df = self.mut_all_genes_df.join(sample_id_df, how='inner')

            all_data = pd.concat([cnv_joined_df, methylation_joined_df, mut_joined_df], 
                                 keys=['cnv', 'methylation', 'mutation'], join='inner', axis=1)
            all_data = all_data.swaplevel(i=0, j=1, axis=1)
            order = all_data.columns.levels[0]
            all_data = all_data.reindex(columns=order, level=0)

            x = all_data.values
            return x
        
        proc = get_processor(self.pre_params)
        if proc:
            x_train = get_feature_array(respone_train_array)
            proc.fit(x_train)

        return proc