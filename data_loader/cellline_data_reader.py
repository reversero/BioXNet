#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from os.path import exists
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold


class CellLineBaseData():
    def __init__(self, data_dir,
                       data_list, 
                       logger,
                       remove_response_outlier=False,
                       mut_binary=False, 
                       selected_genes=None, 
                       combine_type='intersection', 
                       use_coding_genes_only=False):
        self.data_path = Path(data_dir)
        self.logger = logger
        self.remove_response_outlier = remove_response_outlier
        self.data_list = data_list
        self.mut_binry = mut_binary
        self.selected_genes = selected_genes
        self.combine_type = combine_type
        self.use_coding_genes_only = use_coding_genes_only

        data_folder_name = '_'.join(data_list)
        self.prepared_path = Path(data_dir).joinpath(data_folder_name, 'prepared')
        self.processed_path = Path(data_dir).joinpath(data_folder_name, 'processed')
        self.split_path = Path(data_dir).joinpath(data_folder_name, 'split')

        self.get_dataset()

    def get_dataset(self):
        if exists(self.processed_path.joinpath('cnv_allgenes.csv')) and \
            exists(self.processed_path.joinpath('methylation_allgenes.csv')) and \
                exists(self.processed_path.joinpath('mut_allgenes.csv')) and \
                    exists(self.processed_path.joinpath('drug_target_allgenes.csv')) and \
                        exists(self.processed_path.joinpath('response.csv')):
            self.logger.info('Loading processed data...')
            self.recreate = False
            self.response_df = pd.read_csv(self.processed_path.joinpath('response.csv'))
            self.cnv_allgenes_df = pd.read_csv(self.processed_path.joinpath('cnv_allgenes.csv'), index_col=0)
            self.methylation_allgenes_df = pd.read_csv(self.processed_path.joinpath('methylation_allgenes.csv'), index_col=0)
            self.mut_allgenes_df = pd.read_csv(self.processed_path.joinpath('mut_allgenes.csv'), index_col=0)
            self.drug_target_allgenes_df = pd.read_csv(self.processed_path.joinpath('drug_target_allgenes.csv'), index_col=0)
        else:
            self.logger.info('Processing data...\n')
            self.recreate = True
            if not exists(self.processed_path):
                os.makedirs(self.processed_path)

            self.response_df, self.cnv_df, self.methylation_df, \
                self.mutation_df, self.drug_target_df = self._load_data()
            self.sample_list = self._combine_samples()
            self.genes, self.cnv_allgenes_df, self.methylation_allgenes_df, \
                self.mut_allgenes_df, self.drug_target_allgenes_df = self._combine_genes()
            self.response_df = self.response_df[self.response_df['CellLine'].isin(self.sample_list)]

            self.logger.info('Save processed data...\n')
            genes_df = pd.DataFrame(self.genes, columns=['genes'])
            genes_df.to_csv(self.processed_path.joinpath('genes.csv'), index=False)
            self.cnv_allgenes_df.to_csv(self.processed_path.joinpath('cnv_allgenes.csv'))
            self.methylation_allgenes_df.to_csv(self.processed_path.joinpath('methylation_allgenes.csv'))
            self.mut_allgenes_df.to_csv(self.processed_path.joinpath('mut_allgenes.csv'))
            self.drug_target_allgenes_df.to_csv(self.processed_path.joinpath('drug_target_allgenes.csv'))
            self.response_df.to_csv(self.processed_path.joinpath('response.csv'), index=False)

        self.logger.info(f'The shape of cnv_allgenes_df {self.cnv_allgenes_df.shape}')
        self.logger.info(f'The shape of methylation_allgenes_df {self.methylation_allgenes_df.shape}')
        self.logger.info(f'The shape of mutation_allgenes_df {self.mut_allgenes_df.shape}')
        self.logger.info(f'The shape of drug_target_allgenes_df {self.drug_target_allgenes_df.shape}')

        self.logger.info('{} samples and {} drugs in {} records.'.format(
            len(set(self.response_df['CellLine'])), len(set(self.response_df['DrugName'])), len(self.response_df)))

        self.genes = list(self.cnv_allgenes_df.columns)
        self.response_array = self.response_df.values.tolist()

    def get_genes(self):
        return self.genes

    def get_features(self):
        return self.cnv_allgenes_df, self.methylation_allgenes_df, \
               self.mut_allgenes_df, self.drug_target_allgenes_df
    
    def _load_data(self):
        response_df = self._load_response_data()
        cnv_df, methylation_df, mutation_df = self._load_cellline_data()

        drug_target_df = self._load_drug_target()
        drug_with_target_list = list(drug_target_df.index)
        
        response_df = response_df[response_df['DrugName'].isin(drug_with_target_list)]  
        self.logger.info('After keeping the drugs that have targets in Reactome, {} samples and {} drugs in {} records.\n'.format(
            len(set(response_df['CellLine'])), 
            len(set(response_df['DrugName'])), 
            len(response_df)))  
        
        return response_df, cnv_df, methylation_df, mutation_df, drug_target_df
    
    def _load_response_data(self):
        self.logger.info('Loading drug response data.')
        response_data_list = []
        for data in self.data_list:
            if data == 'GDSC':
                ic50_df = pd.read_csv(self.data_path.joinpath('GDSC', 'prepared', 'gdsc_ic50.csv'),
                                      usecols=['CellLine', 'DrugName', 'ln_ic50']) 
                self.logger.info('Change the ln_ic50 in GDSC to pIC50 by negative operation')
                ic50_df['pIC50'] = -ic50_df['ln_ic50']
                # drop the ln_ic50 column
                ic50_df.drop(columns=['ln_ic50'], inplace=True)
                response_data_list.append(ic50_df)
            elif data == 'CTRPv2':
                ic50_df = pd.read_csv(self.data_path.joinpath('CTRPv2', 'prepared', 'ctrpv2_ic50.csv'),
                                      usecols=['CellLine', 'DrugName', 'ic50'])
                self.logger.info('Change the ic50 in CTRPv2 to pIC50')
                ic50_df['pIC50'] = -np.log(ic50_df['ic50'])
                # drop the ic50 column
                ic50_df.drop(columns=['ic50'], inplace=True)
                response_data_list.append(ic50_df)

        response_df = pd.concat(response_data_list)
        self.logger.info(f'The shape of response data {response_df.shape}.')

        if self.remove_response_outlier:
            self.logger.info('Remove the response outlier using IQR.')
            Q1 = response_df['pIC50'].quantile(0.25)
            Q3 = response_df['pIC50'].quantile(0.75)
            IQR = Q3 - Q1
            response_df = response_df[(response_df['pIC50'] >= Q1 - 1.5 * IQR) & (response_df['pIC50'] <= Q3 + 1.5 * IQR)]
            self.logger.info(f'The shape of response data {response_df.shape} after removing outliers.')

        self.logger.info('{} samples and {} drugs in {} records.\n'.format(
            len(set(response_df['CellLine'])), 
            len(set(response_df['DrugName'])), 
            len(response_df)))
        return response_df
    
    def _load_cellline_data(self):
        self.logger.info('Loading cell line multiomics data.')
        def load_one_omics(omics_name):
            omics_df_list = []
            for data in self.data_list:
                if data == 'GDSC':
                    omics_df = pd.read_csv(self.data_path.joinpath('GDSC', 'prepared', omics_name+'.csv'), index_col=0)
                    omics_df_list.append(omics_df)
                    self.logger.info('The shape of {} data in GDSC is {}.'.format(omics_name, omics_df.shape))
                elif data == 'CTRPv2':
                    omics_df = pd.read_csv(self.data_path.joinpath('CTRPv2', 'prepared', omics_name+'.csv'), index_col=0)
                    omics_df_list.append(omics_df)
                    self.logger.info('The shape of {} data in CTRPv2 is {}.'.format(omics_name, omics_df.shape))
            # get the intersection of columns in omics_df_list
            genes_intersection = list(set.intersection(*[set(df.columns) for df in omics_df_list]))
            omics_df_list = [df[genes_intersection] for df in omics_df_list]
            omics_df = pd.concat(omics_df_list)
            # drop the duplicats
            omics_df = omics_df.drop_duplicates()
            self.logger.info(f'The shape of {omics_name} data {omics_df.shape}.')
            return omics_df

        self.logger.info('Loading copy number variation (cnv) data.')
        cnv_df = load_one_omics('cnv')
        self.logger.info('')

        self.logger.info('Loading methylation data.')
        methylation_df = load_one_omics('methylation')
        self.logger.info('')

        self.logger.info('Loading mutation data.')
        mutation_df = load_one_omics('mutation_set_cross_important_only')
        if self.mut_binry:
            self.logger.info('mut_binary = True.\n')
            mutation_df[mutation_df > 1.] = 1.

        return cnv_df, methylation_df, mutation_df

    def _load_drug_target(self):
        self.logger.info('Loading drug-target data.')
        drug_target_df_list = []
        for data in self.data_list:
            if data == 'GDSC':
                drug_target_df = pd.read_csv(self.data_path.joinpath('GDSC', 'prepared', 'drug_target_matrix.csv'), index_col=0)
                self.logger.info('The shape of drug-target data in GDSC is {}.'.format(drug_target_df.shape))
                drug_target_df_list.append(drug_target_df)
            elif data == 'CTRPv2':
                drug_target_df = pd.read_csv(self.data_path.joinpath('CTRPv2', 'prepared', 'drug_target_matrix.csv'), index_col=0)
                self.logger.info('The shape of drug-target data in CTRPv2 is {}.'.format(drug_target_df.shape))
                drug_target_df_list.append(drug_target_df)
        # get the union of columns in drug_target_df_list
        genes_union = list(set.union(*[set(df.columns) for df in drug_target_df_list]))
        all_genes_df = pd.DataFrame(index=genes_union)
        drug_target_df_list = [self.__join_and_fill_missing_values(all_genes_df, df) for df in drug_target_df_list]
        drug_target_df = pd.concat(drug_target_df_list)
        self.logger.info(f'The shape of drug-target data {drug_target_df.shape}.')

        self.logger.info('Only kept the drugs that have targets in Reactome.')
        gene_reactome_df = pd.read_csv(self.data_path.joinpath('reactome', 'genes_level5_roottoleaf.csv'))
        gene_reactome_list = list(set(gene_reactome_df['gene']))
        drug_target_df = drug_target_df[list(set(gene_reactome_list)&set(drug_target_df.columns))]

        self.logger.info('Remove the drugs that have no targets in Reactome.')
        drug_target_df = drug_target_df.loc[~(drug_target_df==0).all(axis=1)]
        self.logger.info(f'The shape of drug-target data {drug_target_df.shape}.\n')

        return drug_target_df

    def _combine_samples(self):
        self.logger.info('Use the intersection of samples')
        sample_list = list(set(self.response_df['CellLine']) & set(self.cnv_df.index) \
                           & set(self.methylation_df.index) & set(self.mutation_df.index))
        self.logger.info(f'In total, {len(sample_list)} samples are included.\n')
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
            coding_genes_df = pd.read_csv(self.data_path.joinpath( 
                'HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt'),
                sep='\t', header=None)
            coding_genes_df.columns = ['chr', 'start', 'end', 'name']
            coding_genes = set(coding_genes_df['name'].unique())
            genes = list(set(genes) & set(coding_genes))
            self.logger.info(f'With using coding genes only, {len(genes)} genes are included.\n')
        
        genes.sort()

        all_genes_df = pd.DataFrame(index=genes)
        cnv_allgenes_df = self.__join_and_fill_missing_values(all_genes_df, self.cnv_df)
        methylation_allgenes_df = self.__join_and_fill_missing_values(all_genes_df, self.methylation_df)
        mutation_allgenes_df = self.__join_and_fill_missing_values(all_genes_df, self.mutation_df)
        drug_target_allgenes_df = self.__join_and_fill_missing_values(all_genes_df, self.drug_target_df)

        return genes, cnv_allgenes_df, methylation_allgenes_df, mutation_allgenes_df, drug_target_allgenes_df

    def __join_and_fill_missing_values(self, all_genes_df, target_df):
        # all the rows from the all_genes_df dataframe will be included in the result, even if there is no matching row in the target_df
        joined_df = target_df.T.join(all_genes_df, how='right')
        joined_df = joined_df.T
        joined_df = joined_df.fillna(0)
        return joined_df

    def train_valid_test_split(self, seed):
        if not exists(self.split_path):
            os.makedirs(self.split_path)
        if not exists(self.split_path.joinpath('training_set_seed_'+str(seed)+'.csv')) or self.recreate == True:
            self.logger.info(f'Spliting the dataset with seed {seed}')
            sample_train, sample_test = train_test_split(self.response_df, test_size=0.2)
            sample_test, sample_valid = train_test_split(sample_test, test_size=0.5)
            sample_train.to_csv(self.split_path.joinpath('training_set_seed_'+str(seed)+'.csv'), index=False)
            sample_valid.to_csv(self.split_path.joinpath('valid_set_seed_'+str(seed)+'.csv'), index=False)
            sample_test.to_csv(self.split_path.joinpath('test_set_seed_'+str(seed)+'.csv'), index=False)
        else:
            self.logger.info(f'Loading the split index with seed {seed}.')
            sample_train = pd.read_csv(self.split_path.joinpath('training_set_seed_'+str(seed)+'.csv'))
            sample_valid = pd.read_csv(self.split_path.joinpath('valid_set_seed_'+str(seed)+'.csv'))
            sample_test = pd.read_csv(self.split_path.joinpath('test_set_seed_'+str(seed)+'.csv'))
        
        self.logger.info(f'{len(sample_train)} samples for train, {len(sample_valid)} samples for valid, {len(sample_test)} samples for test.\n')

        respone_train_array = sample_train.values.tolist()
        respone_valid_array = sample_valid.values.tolist()
        respone_test_array = sample_test.values.tolist()

        return respone_train_array, respone_valid_array, respone_test_array
    
    def train_valid_test_split_Kfold(self, K, seed):
        Kfold_seed_path = self.data_path.joinpath('GDSC', 'Kfold-{}_seed-{}'.format(K, seed))
        if not exists(Kfold_seed_path):
            os.makedirs(Kfold_seed_path)
        if not exists(Kfold_seed_path.joinpath('training_K0.csv')) or self.recreate == True:
            kf = KFold(n_splits=K, shuffle=True, random_state=seed)
            for i, (train_index, test_index) in enumerate(kf.split(self.response_df)):
                sample_train = self.response_df.iloc[train_index]
                sample_test = self.response_df.iloc[test_index]
                sample_train.to_csv(Kfold_seed_path.joinpath('training_K{}.csv'.format(i)), index=False)
                sample_test.to_csv(Kfold_seed_path.joinpath('test_K{}.csv'.format(i)), index=False)
        filename_prefix = ''
        return Kfold_seed_path, filename_prefix


