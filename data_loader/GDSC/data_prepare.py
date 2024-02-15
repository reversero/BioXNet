#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path


class GDSCDataPrepare():
    def __init__(self, logger, data_dir, save_dir, recreate=False):
        # 分析日志 + 保存路径
        self.logger = logger
        self.processed_dir = Path(save_dir).joinpath('prepared')
        if os.path.exists(self.processed_dir):
            self.logger.info(f'{self.processed_dir} exists.')
        else:
            os.makedirs(self.processed_dir, exist_ok=True)
        self.data_dir = Path(data_dir)
        self.recreate = recreate

        # remove silent and intron mutations 过滤突变信息
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

        # 获取数据
        self.selected_cellLine_list = self.prepare_response()
        drug_list = self.prepare_drug_target()
        cnv_cellLine_list = self.prepare_cnv()
        methylation_cellLine_list = self.prepare_methylation450K()
        mutation_cellLine_list = self.prepare_mutation()

        # 保留有所有组学数据的cell line（取交集）
        cellLine_list = list(set(self.selected_cellLine_list) &
                             set(cnv_cellLine_list) &
                             set(methylation_cellLine_list) &
                             set(mutation_cellLine_list))
        cellLine_df = pd.DataFrame({'CellLine': cellLine_list})
        cellLine_df.to_csv(self.processed_dir.joinpath('cellLine_list.csv'), index=False)

    def prepare_cnv(self):
        self.logger.info('Load copy number variation data from CCLE...')
        df = pd.read_csv(self.data_dir.joinpath('CCLE', 'CNV', 'CCLE_copynumber_byGene_2013-12-03.txt'), sep='\t')
        self.logger.info('{} genes have cnv in {} samples.'.format(len(df), len(df.columns)-5))
        df.drop(columns=['EGID', 'CHR', 'CHRLOC', 'CHRLOCEND'], axis=1, inplace=True)
        df.rename(columns={'SYMBOL': 'Gene'}, inplace=True)
        self.logger.info(f'{df.shape} data records existd in the original file.')

        # process the cell line columns
        cellLine_column_list = list(set(df.columns) - set(['Gene']))
        cellLine_column2name_dict = self._cellLine_process(cellLine_column_list)
        df.rename(columns=cellLine_column2name_dict, inplace=True)
        # filter the cell lines
        keep_columns = ['Gene'] + list(set(self.selected_cellLine_list) & set(df.columns))
        df_filter = df[keep_columns]
        self.logger.info(f'{df_filter.shape} data records left after keeping the selected cell lines.')
        del df

        self.logger.info('Thresholded the cnv values to -2, -1, 0, 1, 2')
        cellLine_columns = list(set(df_filter.columns) - set(['Gene']))
        df_filter[cellLine_columns] = df_filter[cellLine_columns].applymap(lambda x: -2 if x <= -1.2 else 
                                                                           (-1 if -1.2 < x <= -0.6 else 
                                                                            (0 if -0.6 < x <= 0.4 else
                                                                             (1 if 0.4 < x <= 0.75 else 2))))
        cnv_array = df_filter[cellLine_columns].values
        self.logger.info('Number of -2, -1, 0, 1, 2: {}, {}, {}, {}, {}'.format(
            (cnv_array == -2).sum(), 
            (cnv_array == -1).sum(), 
            (cnv_array == 0).sum(), 
            (cnv_array == 1).sum(), 
            (cnv_array == 2).sum()))
        del cnv_array

        self.logger.info('Remove the genes with all zero values')
        df_filter = df_filter.loc[~(df_filter==0).all(axis=1)]
        self.logger.info(f'After filter, {df_filter.shape} data records left.')
        
        self.logger.info('Let the cell lines as rows and genes as columns')
        df_filter = df_filter.T
        df_filter.columns = df_filter.iloc[0]
        df_filter.drop(df_filter.index[0], inplace=True)
        df_filter['CellLine'] = df_filter.index
        keep_columns = ['CellLine'] + list(set(df_filter.columns) - set(['CellLine']))
        df_filter = df_filter[keep_columns]
        # save
        df_filter.to_csv(self.processed_dir.joinpath('cnv.csv'), index=False)
        self.logger.info('After processing, {} genes have cnv in {} cell lines.\n'.format(len(df_filter.columns)-1,
            len(df_filter)))
        return list(df_filter['CellLine'])

    def prepare_methylation450K(self):
        self.logger.info('Load methylation 450K data...')
        if os.path.exists(self.processed_dir.joinpath('methylation.csv')) and not self.recreate:
            self.logger.info(f'{self.processed_dir.joinpath("methylation.csv")} exists.')
            df = pd.read_csv(self.processed_dir.joinpath('methylation.csv'))
            cellLine_list = list(set(df['CellLine']))
            self.logger.info('{} cell lines have been selected in Methylation.\n'.format(len(cellLine_list)))
            return cellLine_list

        df = pd.read_csv(self.data_dir.joinpath('GDSC', 'raw', 'methylation_450K_processed.csv'))
        
        # process the cell line columns
        cellLine_column_list = list(set(df.columns) - set(['gene']))
        cellLine_column2name_dict = {cellline_column: cellline_column.replace('-', '').upper() for cellline_column in cellLine_column_list}
        df.rename(columns=cellLine_column2name_dict, inplace=True)
        self.logger.info(f'{df.shape} data records in the original file.')

        # filter the cell lines
        keep_columns = ['gene'] + list(set(self.selected_cellLine_list) & set(df.columns))
        df_filter = df[keep_columns]
        self.logger.info(f'{df_filter.shape} records left after keeping the selected cell lines.')
        del df
        
        self.logger.info('let the cell lines as rows and genes as columns')
        df_filter = df_filter.T
        df_filter.columns = df_filter.iloc[0]
        df_filter.drop(df_filter.index[0], inplace=True)
        df_filter['CellLine'] = df_filter.index
        keep_columns = ['CellLine'] + list(set(df_filter.columns) - set(['CellLine']))
        df_filter = df_filter[keep_columns]
        self.logger.info(f'Current shape {df_filter.shape}')

        self.logger.info('Drop nan values')
        df_filter.dropna(axis=0, how='any', inplace=True)
        self.logger.info(f'After drop nan, {df_filter.shape} records left.')

        # save
        df_filter.to_csv(self.processed_dir.joinpath('methylation.csv'), index=False)
        self.logger.info('After processing, {} genes have methylation in {} cell lines.\n'.format(len(df_filter.columns)-1,
            len(df_filter)))
        return list(df_filter['CellLine'])

    def prepare_methylation(self):
        self.logger.info('Load methylation data...')
        df = pd.read_csv(self.data_dir.joinpath('CCLE', 'methylation', 
            'CCLE_RRBS_TSS_1kb_20180614.txt'), sep='\t')
        
        # process the cell line columns
        cellLine_column_list = list(set(df.columns) - set(['TSS_id', 'gene', 'chr', 'fpos', 'strand', 'avg_coverage']))
        cellLine_column2name_dict = self._cellLine_process(cellLine_column_list)
        df.rename(columns=cellLine_column2name_dict, inplace=True)
        self.logger.info(f'{df.shape} data records in the original file.')

        # filter the cell lines
        keep_columns = ['gene'] + list(set(self.selected_cellLine_list) & set(df.columns))
        df_filter = df[keep_columns]
        self.logger.info(f'{df_filter.shape} records left after keeping the selected cell lines.')
        del df

        self.logger.info("Replace '     NA' with 0")
        df_filter.replace('     NA', 0, inplace=True)
        
        self.logger.info('let the cell lines as rows and genes as columns')
        df_filter = df_filter.T
        df_filter.columns = df_filter.iloc[0]
        df_filter.drop(df_filter.index[0], inplace=True)
        df_filter['CellLine'] = df_filter.index
        keep_columns = ['CellLine'] + list(set(df_filter.columns) - set(['CellLine']))
        df_filter = df_filter[keep_columns]
        self.logger.info(f'Current shape {df_filter.shape}')

        self.logger.info('Drop nan values')
        df_filter.dropna(axis=0, how='any', inplace=True)
        self.logger.info(f'After drop nan, {df_filter.shape} records left.')

        # save
        df_filter.to_csv(self.processed_dir.joinpath('methylation.csv'), index=False)
        self.logger.info('After processing, {} genes have methylation in {} cell lines.\n'.format(len(df_filter.columns)-1,
            len(df_filter)))
        return list(df_filter['CellLine'])

    def prepare_mutation(self):
        self.logger.info('Load mutation data...')
        df = pd.read_csv(self.data_dir.joinpath('CCLE', 'mutation', 'OmicsSomaticMutations.csv'),
                         usecols=['HugoSymbol', 'VariantInfo', 'DepMap_ID'])
        
        self.logger.info('Mapping the DepMap_ID to the cell line names')
        depMap2cellLine_df = pd.read_csv(self.data_dir.joinpath('CCLE', 'resources', 'Model.csv'))
        depMap2CellLine_dict = dict(zip(depMap2cellLine_df['ModelID'], depMap2cellLine_df['StrippedCellLineName']))
        df['CellLine'] = df['DepMap_ID'].map(depMap2CellLine_dict)
        df_filter = df[df['CellLine'].isin(self.selected_cellLine_list)]
        self.logger.info(f'{df_filter.shape} records left after keeping the selected cell lines.')
        del df
        
        # process the variation classification        
        if self.filter_silent_muts:
            df_filter = df_filter[df_filter['VariantInfo'] != 'SILENT']
            self.logger.info('Excluding the VariantInfo as SILENT')
        if self.filter_missense_muts:
            df_filter = df_filter[df_filter['VariantInfo'] != 'MISSENSE']
            self.logger.info('Excluding the VariantInfo as MISSENSE')
        if self.filter_introns_muts:
            df_filter = df_filter[df_filter['VariantInfo'] != 'Intron']
            self.logger.info('Excluding the VariantInfo as Intron')

        # important_only = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Splice_Site','Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Start_Codon_SNP','Nonstop_Mutation', 'De_novo_Start_OutOfFrame', 'De_novo_Start_InFrame']
        exclude = ['SILENT', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
        if self.keep_important_only:
            # remove the variants in exclude
            df_filter = df_filter[~df_filter['VariantInfo'].isin(exclude)]
            self.logger.info(f'Excluding the VariantInfo as {exclude}')
        if self.truncating_only:
            include = ['NONSENSE', 'FRAME_SHIFT_DEL', 'FRAME_SHIFT_INS']
            df_filter = df_filter[df_filter['VariantInfo'].isin(include)]
            self.logger.info(f'Only keeping the VariantInfo as {include}')
        
        # the columns of the rearange table are the genes, the rows are the samples and the values denote that whether one sample has this mutation on this gene
        df_table = pd.pivot_table(data=df_filter, index='CellLine', columns='HugoSymbol', 
                                  values='VariantInfo', aggfunc='count')
        df_table = df_table.fillna(0)
        total_numb_mutations = df_table.sum().sum()

        number_samples = df_table.shape[0]
        self.logger.info(f'number of mutations {total_numb_mutations}, {total_numb_mutations / (number_samples + 0.0)}')
        self.logger.info('After processing, {} genes have methylation in {} cell lines.'.format(
            len(df_table.columns)-1, len(df_table)))
        filename = self.processed_dir.joinpath('mutation_set_cross_' + self.ext + '.csv')
        df_table.to_csv(filename)

        return list(df_table.index)

    def prepare_drug_target(self):
        self.logger.info('Load drug target data...')
        df = pd.read_csv(self.data_dir.joinpath('GDSC', 'drug', 'drug_target.csv'))
        self.logger.info(f'{df.shape} drug records in the original file.')

        self.logger.info('Let the drugs be rows and the targets be columns -> a matrix')
        df['drug_name'] = df['drug_name'].str.lower()
        # the columns of the rearange table are the genes, the rows are the drugs and the values denote that whether one drug has this target
        df_table = pd.crosstab(df['drug_name'], df['Gene'])
        # remove index name and column name
        df_table.index.name = None
        df_table.columns.name = None
        self.logger.info('After processing, the shape of drug_target data is {}.\n'.format(df_table.shape))

        df_table.to_csv(self.processed_dir.joinpath('drug_target_matrix.csv'))
        return list(df_table.index)

    def prepare_response(self):
        self.logger.info('Load drug data...')
        gdsc1_df = pd.read_excel(self.data_dir.joinpath('GDSC', 'raw', 'GDSC1_fitted_dose_response_24Jul22.xlsx'))
        gdsc2_df = pd.read_excel(self.data_dir.joinpath('GDSC', 'raw', 'GDSC2_fitted_dose_response_24Jul22.xlsx'))
        keep_columns = ['DATASET', 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']
        gdsc1_df = gdsc1_df[keep_columns]
        gdsc2_df = gdsc2_df[keep_columns]
        gdsc1_df = gdsc1_df.drop_duplicates()
        gdsc2_df = gdsc2_df.drop_duplicates()
        gdsc_df = pd.concat([gdsc1_df, gdsc2_df], ignore_index=True)
        df = gdsc_df.drop_duplicates()
        df = df[['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']]
        self.logger.info(f'{df.shape} drug records in the original file.')

        self.logger.info('filter by keeping the data with valid drug_name and IC50 values')
        df_filter = df[(df['DRUG_NAME'].notna()) & (df['LN_IC50'].notna())]
        df_filter['DRUG_NAME'] = df_filter['DRUG_NAME'].str.lower()
        self.logger.info(f'After filtering, {df_filter.shape} drug records left.')

        self.logger.info('Remove the hyphens in the cell line names')
        df_filter['CELL_LINE_NAME'] = df_filter['CELL_LINE_NAME'].str.replace('-', '')
        df_filter['CELL_LINE_NAME'] = df_filter['CELL_LINE_NAME'].str.upper()
        
        self.logger.info('filter by the drug-target info')
        drug_target_df = pd.read_csv(self.data_dir.joinpath('GDSC', 'drug', 'drug_target.csv'))
        drug_target_df['drug_name'] = drug_target_df['drug_name'].str.lower()
        drug_name_list = list(set(drug_target_df['drug_name']))
        self.logger.info('only keep the drugs with target info')
        drug_name_list = list(set(drug_target_df['drug_name']))
        df_filter = df_filter[df_filter['DRUG_NAME'].isin(drug_name_list)]
        df_filter.dropna(inplace=True)
        self.logger.info(f'After filtering, {df_filter.shape} drug records left.')

        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After removing duplicates, {df_filter.shape} drug records left.')

        self.logger.info('Sort the IC50 data by the cell line names')
        df_filter = df_filter.sort_values(by=['CELL_LINE_NAME'])

        '''save'''
        df_filter.columns = ['CellLine', 'DrugName', 'ln_ic50']
        df_filter.to_csv(self.processed_dir.joinpath('gdsc_ic50.csv'), index=False)
        cellLine_list = list(set(df_filter['CellLine']))
        self.logger.info('{} cell lines have been selected.\n'.format(len(cellLine_list)))
        return cellLine_list

    def _cellLine_process(self, cellLine_column_list):
        '''
        The cell lines in CNV and methylation data are in the format of LOUNH91_LUNG,
        we need to remove the suffixes to match the cell lines in mutation data.
        Note that for the repeated cell lines, we will remove them.
        Args:
            cellLine_list (list):
        '''
        cellLine_split_list = [x.split('_')[0] for x in cellLine_column_list]
        cellLine_column2name_dict = dict()
        for cellLine_column in cellLine_column_list:
            cellLine = cellLine_column.split('_')[0]
            if cellLine_split_list.count(cellLine) > 1:
                continue
            cellLine_column2name_dict[cellLine_column] = cellLine
        return cellLine_column2name_dict


import logging

# 设置日志记录的配置
logging.basicConfig(level=logging.DEBUG, filename='GDSC_Data_Prepare.log', filemode='w',
                    format="%(asctime)s - %(name)s - %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger('TCGA_Data')

MyTCGADataPrepare = GDSCDataPrepare(logger=logger, 
                                    data_dir='../../data',
                                    save_dir='../../data/GDSC')