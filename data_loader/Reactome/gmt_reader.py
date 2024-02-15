import os
import re
import pandas as pd

'''
    处理GMT（Gene Matrix Transposed）格式的文件
'''

# data_dir = os.path.dirname(__file__)
class GMT():
    # genes_cols : start reading genes from genes_col(default 1, it can be 2 e.g. if an information col is added after the pathway col)
    # pathway col is considered to be the first column (0)
    
    '''
        从GMT文件中加载数据到DataFrame：
            - 打开GMT文件并逐行读取数据
            - 对于每一行数据，先用strip()方法去除两端的空白符，然后用split('\t')方法根据制表符分割成基因列表
            - 对于每个基因，根据指定的genes_col和pathway_col索引获取通路和基因名称，并存储为字典
            - 将所有字典组成的列表转换为DataFrame
            - 返回生成的DataFrame对象
    '''
    def load_data(self, filename, genes_col=1, pathway_col=0):
        data_dict_list = []
        with open(filename) as gmt:
            data_list = gmt.readlines()
            # print data_list[0]
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)
        # print df.head()

        return df


    '''
        从GMT文件中加载数据到一个字典：
            - 打开GMT文件并逐行读取数据
            - 对于每一行数据，先用split('\t')方法根据制表符分割成基因列表
            - 使用列表的第一个元素作为键，将其余部分作为值存储到字典中
            - 返回包含键值对的字典对象
    '''
    def load_data_dict(self, filename):
        data_dict_list = []
        dict = {}
        with open(filename) as gmt:
            data_list = gmt.readlines()
            # print data_list[0]
            for row in data_list:
                genes = row.split('\t')
                dict[genes[0]] = genes[2:]

        return dict


    '''
        将字典中的数据写入到GMT文件中：
            - 创建一个空列表lines，用于存储要写入文件的行
            - 使用open函数以写入模式打开指定路径的GMT文件
            - 遍历字典中的键值对：
                - 对于每个键值对，将值转换为字符串，并使用join方法以制表符分隔成一个字符串str1
                - 将键和str1拼接成一行字符串line，并添加到lines列表中
            - 使用writelines方法将lines列表中的所有行写入到GMT文件中
            - 返回，方法执行完毕
    '''
    def write_dict_to_file(self, dict, filename):
        lines = []
        with open(filename, 'w') as gmt:
            for k in dict:
                str1 = '	'.join(str(e) for e in dict[k])
                line = str(k) + '	' + str1 + '\n'
                lines.append(line)
            gmt.writelines(lines)
        return
