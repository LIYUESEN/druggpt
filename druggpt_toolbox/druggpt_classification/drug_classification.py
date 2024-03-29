# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:40:31 2023

@author: Sen
"""

#%%网络代理
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default=None, help='Input the dirpath')
parser.add_argument('-f', type=str, default=None, help='Input the FASTA file')
args = parser.parse_args()

dirpath = args.d
fasta_filepath = args.f

if dirpath[-1] != '/' :
    dirpath = dirpath + '/'
#%%加载数据集
from datasets import load_dataset
dataset = load_dataset("jglaser/binding_affinity")['train']
#%%加载生成的药物
import pandas as pd
dirpath = dirpath
mapping_data = pd.read_csv(os.path.join(dirpath,'hash_ligand_mapping.csv'), header=None)
generate_drug_list =  mapping_data.iloc[:, 1].tolist()

#%%目标蛋白加载fasta文件
fasta_filepath = fasta_filepath
def read_fasta_file(file_path):
    with open(file_path, 'r') as fasta_file:
        sequence = []
        
        for line in fasta_file:
            line = line.strip()
            if not line.startswith('>'):
                sequence.append(line)

        protein_sequence = ''.join(sequence)

    return protein_sequence
protein_seq = read_fasta_file(fasta_filepath)
#%%检索已经针对目标蛋白存在的药物
drug_exist_dataset = dataset.filter(lambda x:(x["smiles"] in generate_drug_list) and
                              (x['seq'] == protein_seq))
drug_exist_list = list(set(drug_exist_dataset['smiles']))
#%%检索老分子新用途的药物
drug_exist_other_dataset = dataset.filter(lambda x:(x["smiles"] in generate_drug_list) and 
                              (x['seq'] != protein_seq))
drug_exist_other_list = list(set(drug_exist_other_dataset['smiles']))
drug_exist_other_list = [x for x in drug_exist_other_list 
                         if x not in drug_exist_list]
#药物原来的靶标这一步现在没有必要
#因为目前还没有方法筛选掉这些分子是否还作用于老靶点
#通过先移动走作用于老靶点的分子即可实现，或者在这里做一个差集

#%%
def smiles_to_filename(mapping_data,smiles_list):
    #该函数只能针对没有重复元素的smiles列表
    data = mapping_data
    data_tmp = data[data.iloc[:, 1].isin(smiles_list)]
    filename_list = data_tmp.iloc[:, 0].tolist()
    filename_list = [x+'.sdf' for x in filename_list]
    return filename_list

drug_exist_filename_list = smiles_to_filename(mapping_data,drug_exist_list)
drug_exist_other_filename_list = smiles_to_filename(mapping_data,drug_exist_other_list)



#%%
import os 
import shutil

def create_directory(dir_name):
    # 使用os.path.exists()检查目录是否存在
    if not os.path.exists(dir_name):
        # 如果目录不存在，使用os.makedirs()创建它
        os.makedirs(dir_name)

create_directory(os.path.join(dirpath,'drug_exist'))
create_directory(os.path.join(dirpath,'drug_other'))

for filename in drug_exist_filename_list:
    source_filepath = os.path.join(dirpath,filename)
    destination_path = os.path.join(dirpath,'drug_exist',filename)
    try:
        shutil.move(source_filepath,destination_path)
    except FileNotFoundError:
        pass


for filename in drug_exist_other_filename_list:
    source_filepath = os.path.join(dirpath,filename)
    destination_path = os.path.join(dirpath,'drug_other',filename)
    try:
        shutil.move(source_filepath,destination_path)
    except FileNotFoundError:
        pass

create_directory(os.path.join(dirpath,'drug_new'))
for filename in os.listdir(dirpath):
    if filename[-4:] == '.sdf':
        source_filepath = os.path.join(dirpath,filename)
        destination_path = os.path.join(dirpath,'drug_new',filename)
        try:
            shutil.move(source_filepath,destination_path)
        except FileNotFoundError:
            pass

#%%copy hash_ligand_mapping.csv
source_filepath = os.path.join(dirpath, 'hash_ligand_mapping.csv')
shutil.copy(source_filepath,os.path.join(dirpath,'drug_exist', 'hash_ligand_mapping.csv'))
shutil.copy(source_filepath,os.path.join(dirpath,'drug_other', 'hash_ligand_mapping.csv'))
shutil.copy(source_filepath,os.path.join(dirpath,'drug_new', 'hash_ligand_mapping.csv'))

#%%
other_drug_dataset = dataset.filter(lambda x:(x["smiles"] in drug_exist_other_list))
data = mapping_data
other_drug_fliename_list = [data[data.iloc[:, 1]==x].iloc[:, 0].values[0] + '.sdf'
                 for x in other_drug_dataset['smiles']]

other_drug_df = pd.DataFrame({
    'filename': other_drug_fliename_list,
    'smiles': other_drug_dataset['smiles'],
    'seq': other_drug_dataset['seq']   
})

other_drug_df.to_csv(os.path.join(dirpath,'drug_other','otherdrug.csv'), index=False)
    
#%%rename
# 原文件夹名称
dir_name = os.path.basename(os.path.dirname(dirpath))
dir_path = os.path.dirname(os.path.dirname(dirpath))

original_folder_name  = os.path.join(dir_path,dir_name)

original_drug_exist_name  = os.path.join(dir_path,dir_name,'drug_exist')
original_drug_other_name  = os.path.join(dir_path,dir_name,'drug_other')
original_drug_new_name  = os.path.join(dir_path,dir_name,'drug_new')

new_drug_exist_name  = os.path.join(dir_path,dir_name,dir_name + '_drug_exist')
new_drug_other_name  = os.path.join(dir_path,dir_name,dir_name + '_drug_other')
new_drug_new_name  = os.path.join(dir_path,dir_name,dir_name + '_drug_new')

os.rename(original_drug_exist_name , new_drug_exist_name )
os.rename(original_drug_other_name , new_drug_other_name )
os.rename(original_drug_new_name , new_drug_new_name )
# 新文件夹名称
new_folder_name = os.path.join(dir_path,dir_name+'_class')

# 使用os.rename()函数重命名文件夹
os.rename(original_folder_name, new_folder_name)

























