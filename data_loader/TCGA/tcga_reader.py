#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from os.path import join, exists
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import class_weight


class TCGAData():
    def __init__(self, data_dir, 
                       data_list,
                       logger,
                       mut_binary=False, 
                       selected_genes=None, 
                       combine_type='intersection', 
                       use_coding_genes_only=False,
                       binary_response=True):
        self.data_path = data_dir
        self.logger = logger
        self.data_list = data_list
        self.tcga_data_path = join(data_dir, 'TCGA', 'prepared')
        self.tcga_processed_path = join(data_dir, 'TCGA', 'processed')
        self.mut_binry = mut_binary
        self.selected_genes = selected_genes
        self.combine_type = combine_type
        self.use_coding_genes_only = use_coding_genes_only
        self.binary_response = binary_response

        if exists(join(self.tcga_processed_path, 'cnv_allgenes.csv')) and \
            exists(join(self.tcga_processed_path, 'methylation_allgenes.csv')) and \
                exists(join(self.tcga_processed_path, 'mut_allgenes.csv')) and \
                    exists(join(self.tcga_processed_path, 'drug_target_allgenes.csv')) and \
                        exists(join(self.tcga_processed_path, 'response.csv')):
            self.logger.info('Loading processed data...')
            self.recreate = False
            self.response_df = pd.read_csv(join(self.tcga_processed_path, 'response.csv'))
            self.cnv_allgenes_df = pd.read_csv(join(self.tcga_processed_path, 'cnv_allgenes.csv'), index_col=0)
            self.methylation_allgenes_df = pd.read_csv(join(self.tcga_processed_path, 'methylation_allgenes.csv'), index_col=0)
            self.mut_allgenes_df = pd.read_csv(join(self.tcga_processed_path, 'mut_allgenes.csv'), index_col=0)
            self.drug_target_allgenes_df = pd.read_csv(join(self.tcga_processed_path, 'drug_target_allgenes.csv'), index_col=0)
        else:
            self.logger.info('Processing data...')
            self.recreate = True
            if not exists(self.tcga_processed_path):
                os.makedirs(self.tcga_processed_path)
            self._load_data()
            self.sample_list = self._combine_samples()
            self.genes, self.cnv_allgenes_df, self.methylation_allgenes_df, \
                self.mut_allgenes_df, self.drug_target_allgenes_df = self._combine_genes()
            self.response_df = self.response_df[self.response_df['bcr_patient_barcode'].isin(self.sample_list)]

            self.logger.info('Save processed data...\n')
            self.genes_df = pd.DataFrame(self.genes, columns=['genes'])
            self.genes_df.to_csv(join(self.tcga_processed_path, 'genes.csv'), index=False)
            self.cnv_allgenes_df.to_csv(join(self.tcga_processed_path, 'cnv_allgenes.csv'))
            self.methylation_allgenes_df.to_csv(join(self.tcga_processed_path, 'methylation_allgenes.csv'))
            self.mut_allgenes_df.to_csv(join(self.tcga_processed_path, 'mut_allgenes.csv'))
            self.drug_target_allgenes_df.to_csv(join(self.tcga_processed_path, 'drug_target_allgenes.csv'))
            self.response_df.to_csv(join(self.tcga_processed_path, 'response.csv'), index=False)

        self.logger.info(f'The shape of cnv_allgenes_df {self.cnv_allgenes_df.shape}')
        self.logger.info(f'The shape of methylation_allgenes_df {self.methylation_allgenes_df.shape}')
        self.logger.info(f'The shape of mutation_allgenes_df {self.mut_allgenes_df.shape}')
        self.logger.info(f'The shape of drug_target_allgenes_df {self.drug_target_allgenes_df.shape}')

        self.logger.info('{} samples and {} drugs in {} records.'.format(
            len(set(self.response_df['bcr_patient_barcode'])), len(set(self.response_df['drug_drugbankid'])), len(self.response_df)))

        self.genes = list(self.cnv_allgenes_df.columns)
        self.response_array = self.response_df.values.tolist()

    def get_genes(self):
        return self.genes

    def get_features(self):
        return self.cnv_allgenes_df, self.methylation_allgenes_df, \
               self.mut_allgenes_df, self.drug_target_allgenes_df

    def _load_data(self):
        self.logger.info('Loading response data.')
        response_df = pd.read_csv(join(self.tcga_data_path, 'tcga_drug_response.csv'),
                                  usecols=['bcr_patient_barcode', 'drug_drugbankid', 'drug_response'])
        self.logger.info(f'The shape of response data {response_df.shape}.')
        self.logger.info('{} samples and {} drugs in {} records.'.format(
            len(set(response_df['bcr_patient_barcode'])), 
            len(set(response_df['drug_drugbankid'])), 
            len(response_df)))
        self.response_df = response_df

        self.logger.info('Loading copy number variation (cnv) data.')
        cnv_df = pd.read_csv(join(self.tcga_data_path, 'cnv.csv'), index_col=0)
        self.logger.info(f'The shape of cnv data {cnv_df.shape}.\n')
        self.cnv_df = cnv_df

        self.logger.info('Loading methylation data.')
        methylation_df = pd.read_csv(join(self.tcga_data_path, 'methylation.csv'))
        self.logger.info(f'The shape of methylation data {methylation_df.shape}.')
        self.logger.info('Since there are duplicated samples in methylation data, we will compute mean values.')
        methylation_df = methylation_df.groupby(['Sample']).mean()
        self.logger.info(f'The shape of methylation data {methylation_df.shape}.\n')
        self.methylation_df = methylation_df

        self.logger.info('Loading mutation data.')
        mutation_df = pd.read_csv(join(self.tcga_data_path, 'mutation_set_cross_important_only.csv'),
                                    index_col=0)
        if self.mut_binry:
            self.logger.info('mut_binary = True')
            mutation_df[mutation_df > 1.] = 1.
        self.logger.info(f'The shape of mutation data {mutation_df.shape}.\n')
        self.mutation_df = mutation_df

        self.logger.info('Loading drug-target data.')
        drug_target_df = pd.read_csv(join(self.tcga_data_path, 'drug_target_matrix.csv'), index_col=0)
        self.logger.info(f'The shape of drug-target data {drug_target_df.shape}.\n')
        self.drug_target_df = drug_target_df
        
        self.logger.info('Only kept the drugs that have targets in Reactome.')
        gene_reactome_df = pd.read_csv(join(self.data_path, 'reactome', 'genes_level5_roottoleaf.csv'))
        gene_reactome_list = list(set(gene_reactome_df['gene']))
        self.drug_target_df = self.drug_target_df[list(set(gene_reactome_list)&set(self.drug_target_df.columns))]
        self.logger.info('Remove the drugs that have no targets in Reactome.')
        self.drug_target_df = self.drug_target_df.loc[~(self.drug_target_df==0).all(axis=1)]
        self.logger.info(f'The shape of drug-target data {self.drug_target_df.shape}.\n')
        drug_with_target_list = list(self.drug_target_df.index)
        self.response_df = self.response_df[self.response_df['drug_drugbankid'].isin(drug_with_target_list)]  
        self.logger.info('After keeping the drugs that have targets in Reactome, {} samples and {} drugs in {} records.\n'.format(
            len(set(self.response_df['bcr_patient_barcode'])), 
            len(set(self.response_df['drug_drugbankid'])), 
            len(self.response_df)))    

    def _combine_samples(self):
        self.logger.info('Use the intersection of samples')
        sample_list = list(set(self.response_df['bcr_patient_barcode']) & set(self.cnv_df.index) \
                           & set(self.methylation_df.index) & set(self.mutation_df.index))
        self.logger.info(f'In total, {len(sample_list)} samples are included.')
        return sample_list

    def _combine_genes(self):
        self.logger.info(f'Use the combine type as {self.combine_type} for genes.')
        drug_genes = list(set(self.drug_target_df.columns))
        if self.combine_type == 'intersection':
            genes = list(set(self.cnv_df.columns) & set(self.methylation_df.columns) \
                         & set(self.mutation_df.columns))
            genes = list(set(genes) | set(drug_genes))
        else:
            genes = list(set(self.cnv_df.columns) | set(self.methylation_df.columns) \
                         | set(self.mutation_df.columns) | set(drug_genes))
        self.logger.info(f'In total, {len(genes)} genes are included')

        if self.use_coding_genes_only:
            self.logger.info(f'Use the coding genes from HUGO')
            coding_genes_df = pd.read_csv(join(self.data_path, 
                'HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt'),
                sep='\t', header=None)
            coding_genes_df.columns = ['chr', 'start', 'end', 'name']
            coding_genes = set(coding_genes_df['name'].unique())
            genes = list(set(genes) & set(coding_genes))
            self.logger.info(f'With using coding genes only, {len(genes)} genes are included.')
        
        genes.sort()

        all_genes_df = pd.DataFrame(index=genes)
        cnv_allgenes_df = self.__join_gene(all_genes_df, self.cnv_df)
        methylation_allgenes_df = self.__join_gene(all_genes_df, self.methylation_df)
        mutation_allgenes_df = self.__join_gene(all_genes_df, self.mutation_df)
        drug_target_allgenes_df = self.__join_gene(all_genes_df, self.drug_target_df)

        return genes, cnv_allgenes_df, methylation_allgenes_df, mutation_allgenes_df, drug_target_allgenes_df

    def __join_gene(self, all_genes_df, target_df):
        joined_df = target_df.T.join(all_genes_df, how='right')
        joined_df = joined_df.T
        joined_df = joined_df.fillna(0)
        return joined_df

    def _get_class_num(self, df, data='train'):
        self.logger.info(f'The number of samples for each response value in {data}:')
        for response_value in set(df['drug_response']):
            self.logger.info(f'{response_value}: {len(df[df["drug_response"]==response_value])}')

    def get_class_weights(self, response_train_array):
        if self.binary_response == True:
            self.logger.info('Create class weights for binary response.')
        else:
            self.logger.info('Create class weights for multi-class response.')
        
        label_list = [item[2] for item in response_train_array]
        class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                          classes=np.unique(label_list), 
                                                          y=label_list)
        self.logger.info('Class weights: {} for {}'.format(class_weights, np.unique(label_list)))

        return class_weights

    def train_valid_test_split(self, seed):
        split_dir = join(self.data_path, 'TCGA', 'split')
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        if self.binary_response == True:
            train_file_name = join(split_dir, 'training_set_binary_seed_'+str(seed)+'.csv')
            valid_file_name = join(split_dir, 'validation_set_binary_seed_'+str(seed)+'.csv')
            test_file_name = join(split_dir, 'test_set_binary_seed_'+str(seed)+'.csv')
            self.logger.info('Convert response to binary: clinical response value larger than (or equal) 2 -> 1.')
            self.response_df['drug_response'] = self.response_df['drug_response'].apply(lambda x: 1 if x >= 2 else 0)
            self._get_class_num(self.response_df, data='all')
        else:
            # count the number of samples for each response value
            self._get_class_num(self.response_df, data='all')
            self.logger.info('\n')
            train_file_name = join(split_dir, 'training_set_seed_'+str(seed)+'.csv')
            valid_file_name = join(split_dir, 'valid_set_seed_'+str(seed)+'.csv')
            test_file_name = join(split_dir, 'test_set_seed_'+str(seed)+'.csv')
       
        if not exists(train_file_name) or self.recreate:
            self.logger.info(f'Spliting the dataset with seed {seed}')
            sample_train, sample_test = train_test_split(self.response_df, test_size=0.2)
            sample_test, sample_valid = train_test_split(sample_test, test_size=0.5)
            sample_train.to_csv(train_file_name, index=False)
            sample_valid.to_csv(valid_file_name, index=False)
            sample_test.to_csv(test_file_name, index=False)
        else:
            self.logger.info(f'Loading the split index with seed {seed}.')
            sample_train = pd.read_csv(train_file_name)
            sample_valid = pd.read_csv(valid_file_name)
            sample_test = pd.read_csv(test_file_name)
        
        self._get_class_num(sample_train, data='train')
        self._get_class_num(sample_valid, data='valid')
        self._get_class_num(sample_test, data='test')

        respone_train_array = sample_train.values.tolist()
        respone_valid_array = sample_valid.values.tolist()
        respone_test_array = sample_test.values.tolist()

        return respone_train_array, respone_valid_array, respone_test_array
    
    def train_valid_test_split_Kfold(self, K, seed):
        Kfold_seed_path = Path(self.data_path).joinpath('TCGA', 'Kfold-{}_seed-{}'.format(K, seed))
        if not exists(Kfold_seed_path):
            os.makedirs(Kfold_seed_path)
        if self.binary_response == True:
            self.logger.info('Convert response to binary: clinical response value larger than (or equal) 2 -> 1.')
            self.response_df['drug_response'] = self.response_df['drug_response'].apply(lambda x: 1 if x >= 2 else 0)
            self._get_class_num(self.response_df, data='all')
            filename_prefix = 'binary_'
        else:
            self._get_class_num(self.response_df, data='all')
            filename_prefix = ''

        if not exists(Kfold_seed_path.joinpath(filename_prefix + 'training_K0.csv')) or self.recreate == True:
            kf = KFold(n_splits=K, shuffle=True, random_state=seed)
            for i, (train_index, test_index) in enumerate(kf.split(self.response_df)):
                sample_train = self.response_df.iloc[train_index]
                sample_test = self.response_df.iloc[test_index]
                sample_train.to_csv(Kfold_seed_path.joinpath(filename_prefix + 'training_K{}.csv'.format(i)), index=False)
                sample_test.to_csv(Kfold_seed_path.joinpath(filename_prefix + 'test_K{}.csv'.format(i)), index=False)
        return Kfold_seed_path, filename_prefix

