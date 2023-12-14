#!/usr/bin/env python3
import os, sys
import pandas as pd
from pathlib import Path


class TCGADataPrepare():
    def __init__(self, logger, data_dir, save_dir, recreate):
        self.logger = logger
        self.processed_dir = Path(save_dir).joinpath('prepared')
        if os.path.exists(self.processed_dir):
            self.logger.info(f'{self.processed_dir} exists.')
        else:
            os.makedirs(self.processed_dir, exist_ok=True)
        self.data_dir = Path(data_dir)
        self.recreate = recreate

        # remove silent and intron mutations
        self.filter_silent_muts = False
        self.filter_missense_muts = False
        self.filter_introns_muts = False
        self.keep_important_only = True
        self.truncating_only = False

        self.ext = ""
        if self.keep_important_only:
            self.ext = 'important_only'
        if self.truncating_only:
            self.ext = 'truncating_only'
        if self.filter_silent_muts:
            self.ext = "_no_silent"
        if self.filter_missense_muts:
            self.ext = self.ext + "_no_missense"
        if self.filter_introns_muts:
            self.ext = self.ext + "_no_introns"

        self.selected_sample_list, drug_list = self.prepare_response()
        drug_list = self.prepare_drug_target(drug_list)
        cnv_sample_list = self.prepare_cnv()
        methylation_sample_list = self.prepare_methylation450K()
        mutation_sample_list = self.prepare_mutation()

        sample_list = list(set(self.selected_sample_list) &
                           set(cnv_sample_list) &
                           set(methylation_sample_list) &
                           set(mutation_sample_list))
        sample_df = pd.DataFrame({'Sample': sample_list})
        sample_df.to_csv(self.processed_dir.joinpath('sample_list.csv'), index=False)

    def prepare_cnv(self):
        self.logger.info('Load copy number variation data...')
        df = pd.read_csv(self.data_dir.joinpath('raw', 'all_thresholded.by_genes_whitelisted.tsv'),
            delimiter='\t')
        self.logger.info('{} genes have cnv in {} samples.'.format(len(df), len(df.columns)-3))
        df.drop(columns=['Locus ID', 'Cytoband'], axis=1, inplace=True)
        df.rename(columns={'Gene Symbol': 'Gene'}, inplace=True)
        sample_list = list(set(df.columns) - set(['Gene']))
        self.logger.info(f'{df.shape} data records existd in the original file.')

        # process the barcodes of these samples
        barcode2participant_dict = self.sample_barcode_process(sample_list)
        df.rename(columns=barcode2participant_dict, inplace=True)
        # filter the samples
        keep_columns = ['Gene'] + list(set(self.selected_sample_list) & set(df.columns))
        df_filter = df[keep_columns]
        self.logger.info(f'{df_filter.shape} data records left after keeping the selected samples.')
        del df
        
        self.logger.info('Remove the genes with all zero values')
        df_filter = df_filter.loc[~(df_filter==0).all(axis=1)]
        self.logger.info(f'After filter, {df_filter.shape} data records left.')
        
        self.logger.info('let the samples as rows and genes as columns')
        df_filter = df_filter.T
        df_filter.columns = df_filter.iloc[0]
        df_filter.drop(df_filter.index[0], inplace=True)
        df_filter['Sample'] = df_filter.index
        keep_columns = ['Sample'] + list(set(df_filter.columns) - set(['Sample']))
        df_filter = df_filter[keep_columns]
        # save
        df_filter.to_csv(self.processed_dir.joinpath('cnv.csv'), index=False)
        self.logger.info('After processing, {} genes have cnv in {} samples.\n'.format(len(df_filter.columns)-1,
            len(df_filter)))
        return list(df_filter['Sample'])
    
    def prepare_methylation450K(self):
        self.logger.info('Load methylation 450K data...')
        df = pd.read_csv(self.data_dir.joinpath('raw', 'methylation_450K_processed.csv'))
        # process the barcodes of these samples
        sample_list = list(set(df.columns) - set(['gene']))
        barcode2participant_dict = self.sample_barcode_process(sample_list)
        df.rename(columns=barcode2participant_dict, inplace=True)
        self.logger.info(f'{df.shape} data records in the original file.')

        # filter the samples
        keep_columns = ['gene'] + list(set(self.selected_sample_list) & set(df.columns))
        df_filter = df[keep_columns]
        self.logger.info(f'{df_filter.shape} records left after keeping the selected samples.')
        del df

        self.logger.info('let the samples as rows and genes as columns')
        df_filter = df_filter.T
        df_filter.columns = df_filter.iloc[0]
        df_filter.drop(df_filter.index[0], inplace=True)
        df_filter['Sample'] = df_filter.index
        keep_columns = ['Sample'] + list(set(df_filter.columns) - set(['Sample']))
        df_filter = df_filter[keep_columns]
        self.logger.info(f'Current shape {df_filter.shape}')

        # save
        df_filter.to_csv(self.processed_dir.joinpath('methylation.csv'), index=False)
        self.logger.info('After processing, {} genes have methylation in {} samples.\n'.format(len(df_filter.columns)-1,
            len(df_filter)))
        return list(df_filter['Sample'])

    def prepare_methylation(self):
        self.logger.info('Load methylation data...')
        df = pd.read_csv(self.data_dir.joinpath('TCGA_PanCancer',
            'jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv'), delimiter='\t')
        # process the barcodes of these samples
        sample_list = list(set(df.columns) - set(['Composite Element REF']))
        barcode2participant_dict = self.sample_barcode_process(sample_list)
        df.rename(columns=barcode2participant_dict, inplace=True)
        self.logger.info(f'{df.shape} data records in the original file.')

        # filter the samples
        keep_columns = ['Composite Element REF'] + list(set(self.selected_sample_list) & set(df.columns))
        df_filter = df[keep_columns]
        self.logger.info(f'{df_filter.shape} records left after keeping the selected samples.')
        del df
        
        self.logger.info('map Composite Element REF to gene names')
        cpg_df = pd.read_csv(self.data_dir.joinpath('resources', 'cpg_methylation_cpg_to_annotation.tsv'), delimiter='\t')
        cpg2gene_dict = {row['CpG_id']: row['Symbol'] for _, row in cpg_df.iterrows()}
        df_filter['Composite Element REF'] = df_filter['Composite Element REF'].map(cpg2gene_dict)
        df_filter.rename(columns={'Composite Element REF': 'Gene'}, inplace=True)
        df_filter.dropna(inplace=True)
        self.logger.info(f'{df_filter.shape} data records left after filtering.')
        
        self.logger.info('let the samples as rows and genes as columns')
        df_filter = df_filter.T
        df_filter.columns = df_filter.iloc[0]
        df_filter.drop(df_filter.index[0], inplace=True)
        df_filter['Sample'] = df_filter.index
        keep_columns = ['Sample'] + list(set(df_filter.columns) - set(['Sample']))
        df_filter = df_filter[keep_columns]
        self.logger.info(f'Current shape {df_filter.shape}')

        # save
        df_filter.to_csv(self.processed_dir.joinpath('methylation.csv'), index=False)
        self.logger.info('After processing, {} genes have methylation in {} samples.\n'.format(len(df_filter.columns)-1,
            len(df_filter)))
        return list(df_filter['Sample'])

    def prepare_mutation(self):
        self.logger.info('Load mutation data...')
        df = pd.read_csv(self.data_dir.joinpath('raw', 'mc3.v0.2.8.PUBLIC.maf.gz'), 
                         delimiter='\t', 
                         compression='gzip',
                         usecols=['Hugo_Symbol', 'Variant_Classification', 'Tumor_Sample_Barcode'])
        # process the barcodes of these samples
        sample_list = list(set(df['Tumor_Sample_Barcode']))
        barcode2participant_dict = self.sample_barcode_process(sample_list)
        df['Sample'] = df['Tumor_Sample_Barcode'].map(barcode2participant_dict)
        df_filter = df[df['Sample'].isin(self.selected_sample_list)]
        self.logger.info(f'{df_filter.shape} records left after keeping the selected samples.')
        del df
        
        # process the variation classification        
        if self.filter_silent_muts:
            df_filter = df_filter[df_filter['Variant_Classification'] != 'Silent']
            self.logger.info('Excluding the Variant_Classification as Silent')
        if self.filter_missense_muts:
            df_filter = df_filter[df_filter['Variant_Classification'] != 'Missense_Mutation']
            self.logger.info('Excluding the Variant_Classification as Missense_Mutation')
        if self.filter_introns_muts:
            df_filter = df_filter[df_filter['Variant_Classification'] != 'Intron']
            self.logger.info('Excluding the Variant_Classification as Intron')

        # important_only = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Splice_Site','Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Start_Codon_SNP','Nonstop_Mutation', 'De_novo_Start_OutOfFrame', 'De_novo_Start_InFrame']
        exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
        if self.keep_important_only:
            # remove the variants in exclude
            df_filter = df_filter[~df_filter['Variant_Classification'].isin(exclude)]
            self.logger.info(f'Excluding the Variant_Classification as {exclude}')
        if self.truncating_only:
            include = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']
            df_filter = df_filter[df_filter['Variant_Classification'].isin(include)]
            self.logger.info(f'Only keeping the Variant_Classification as {include}')
        
        # the columns of the rearange table are the genes, the rows are the samples and the values denote that whether one sample has this mutation on this gene
        df_table = pd.pivot_table(data=df_filter, index='Sample', columns='Hugo_Symbol', values='Variant_Classification',
                                  aggfunc='count')
        df_table = df_table.fillna(0)
        total_numb_mutations = df_table.sum().sum()

        number_samples = df_table.shape[0]
        self.logger.info(f'number of mutations {total_numb_mutations}, {total_numb_mutations / (number_samples + 0.0)}')
        self.logger.info('After processing, {} genes have methylation in {} samples.'.format(
            len(df_table.columns)-1, len(df_table)))
        filename = self.processed_dir.joinpath('mutation_set_cross_' + self.ext + '.csv')
        df_table.to_csv(filename)

        return list(df_table.index)

    def prepare_drug_target(self, drug_list):
        self.logger.info('Load drug-target data...')
        df = pd.read_csv(self.data_dir.joinpath('raw', 'drug_target.csv'))
        self.logger.info(f'{df.shape} drug-target records in the original file.')

        self.logger.info('Only keep the drugs that are in the drug response data.')
        df = df[df['drugbank'].isin(drug_list)]
        self.logger.info(f'{df.shape} drug-target records left after keeping the drugs in the drug response data.')
        self.logger.info('{} drugs and {} genes in {} drug-target data.'.format(
            len(set(df['drugbank'])), 
            len(set(df['gene'])),
            len(df)))

        self.logger.info('Let the drugs be row and the targets be columns -> a matrix')
        df_table = pd.crosstab(df['drugbank'], df['gene'])
        # remove index name and column name
        df_table.index.name = None
        df_table.columns.name = None
        self.logger.info('After processing, the shape of drug_target data is {}.\n'.format(df_table.shape))

        df_table.to_csv(self.processed_dir.joinpath('drug_target_matrix.csv'))
        return list(df_table.index)

    def prepare_response(self):
        self.logger.info('Load response data...')
        df = pd.read_csv(self.data_dir.joinpath('raw', 'drug.csv'),
            usecols=['bcr_patient_barcode', 'drug_name', 'measure_of_response'])
        self.logger.info(f'{df.shape} drug records in the original file.')

        self.logger.info('Filter by keeping the data with valid drug_name and measure_of_response')
        df_filter = df[(df['drug_name'].notna()) & (df['measure_of_response'].notna())]
        df_filter['drug_name'] = df_filter['drug_name'].str.lower()
        self.logger.info(f'After filtering, {df_filter.shape} drug records left.')
        
        self.logger.info('Filter by the drug-target info')
        drug_map_df = pd.read_csv(self.data_dir.joinpath('raw', 'drug_mapping.csv'))
        self.logger.info('only keep the drugs with drugbank ids')
        drug_map_df = drug_map_df.dropna(subset=['drugbank'])

        drug_target_df = pd.read_csv(self.data_dir.joinpath('raw', 'drug_target.csv'))
        self.logger.info('Only keep the drugs with target info')
        drug_map_filter_df = drug_map_df[drug_map_df['drugbank'].isin(
            list(set(drug_target_df['drugbank'])))]
        drug_name_list = list(set(drug_map_filter_df['name']))
        drug_name2drugbankid_dict = {row['name']: row['drugbank'] for i, row in drug_map_filter_df.iterrows()}
        
        df_filter = df_filter[df_filter['drug_name'].isin(drug_name_list)]
        df_filter['drug_drugbankid'] = df_filter['drug_name'].map(drug_name2drugbankid_dict)
        df_filter.dropna(inplace=True)
        drug_list = list(set(df_filter['drug_drugbankid']))
        self.logger.info(f'After filtering, {df_filter.shape} drug records left.')

        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After removing duplicates, {df_filter.shape} drug records left.')

        response_map_dir = self.data_dir.joinpath('response_type', 'drug_response.csv')
        self.logger.info(f'Loading drug response type mapping file from {response_map_dir}')
        response_map_df = pd.read_csv(response_map_dir)
        response_map_dict = {row['Response-Type']: row['Map'] for _, row in response_map_df.iterrows()}
        self.logger.info('Map the response to the numeric values')
        df_filter['drug_response'] = df_filter['measure_of_response'].map(response_map_dict)

        self.logger.info('Sort the response data by the barcode')
        df_filter = df_filter.sort_values(by=['bcr_patient_barcode'])

        '''save'''
        df_filter.to_csv(self.processed_dir.joinpath('tcga_drug_response.csv'), index=False)
        sample_list = list(set(df_filter['bcr_patient_barcode']))
        self.logger.info('{} samples have been selected.\n'.format(len(sample_list)))
        return sample_list, drug_list

    def sample_barcode_process(self, sample_list):
        '''
        Given a TCGA-barcode as TCGA-02-0001-01C-01D-0182-01
            TCGA - project
            02 - TSS (Tissue source site), 
            0001 - participant
        Thus we keep the first 3 elements as an unique id for each participant in TCGA.
        Note that the barcode in clinical-drug data is in this format.
        '''
        barcode2participant_dict = dict()
        for barcode in sample_list:
            participant = '-'.join(barcode.split('-')[: 3])
            barcode2participant_dict[barcode] = participant
        return barcode2participant_dict

import logging

logging.basicConfig(level=logging.DEBUG, filename='TCGA_Data_Prepare.log', filemode='w',
                    format="%(asctime)s - %(name)s - %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger('TCGA_Data')

MyTCGADataPrepare = TCGADataPrepare(logger=logger, 
                                    data_dir='../../data/TCGA',
                                    save_dir='../../data/TCGA',
                                    recreate=True)