# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 08:12:43 2023

@author: Sen
"""
#%%
import os 
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default=None, help='Input the dirpath')
args = parser.parse_args()

dirpath = args.d
if dirpath[-1] != '/' :
    dirpath = dirpath + '/'
#%%
def create_directory(dir_name):
    # 使用os.path.exists()检查目录是否存在
    if not os.path.exists(dir_name):
        # 如果目录不存在，使用os.makedirs()创建它
        os.makedirs(dir_name)

input_dirpath = dirpath  #文件夹以/结尾
dir_name = os.path.basename(os.path.dirname(input_dirpath))
dir_path = os.path.dirname(os.path.dirname(input_dirpath))
output_dirpath = os.path.join(dir_path,dir_name+'_min')
create_directory(output_dirpath)

#%%
import openbabel
def sdf_min(input_sdf, output_sdf):
    # 创建一个分子对象
    mol = openbabel.OBMol()

    # 创建转换器，用于文件读写
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("sdf", "sdf")

    # 从SDF文件中读取分子
    conv.ReadFile(mol, input_sdf)

    # 创建力场对象，使用MMFF94力场
    forcefield = openbabel.OBForceField.FindForceField("MMFF94")

    # 为分子设置力场
    success = forcefield.Setup(mol)
    if not success:
        raise Exception("Error setting up force field")

    # 进行能量最小化
    forcefield.SteepestDescent(10000)  # 原来是5000步最速下降法
    forcefield.GetCoordinates(mol)  # 将能量最小化后的坐标保存到分子对象

    # 将能量最小化后的分子写入到SDF文件
    conv.WriteFile(mol, output_sdf)
    
#%% old code
'''
from tqdm import tqdm
file_list = os.listdir(input_dirpath)
for filename in tqdm(file_list):
    if '.sdf' == filename[-4:]:
        input_sdf_file = os.path.join(input_dirpath,filename)
        output_sdf_file = os.path.join(output_dirpath,filename)
        sdf_min(input_sdf_file, output_sdf_file)
    else:
        src_file = os.path.join(input_dirpath,filename)
        dst_file = os.path.join(output_dirpath,filename)
        shutil.copy(src_file,dst_file)#改成copy
'''    

#%% new code
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def handle_file(filename):
    if '.sdf' == filename[-4:]:
        input_sdf_file = os.path.join(input_dirpath, filename)
        output_sdf_file = os.path.join(output_dirpath, filename)
        try:
            sdf_min(input_sdf_file, output_sdf_file)
        except Exception:
            try:
                os.remove(output_sdf_file)
            except Exception:
                print('please remove '+output_sdf_file)
    else:
        src_file = os.path.join(input_dirpath, filename)
        dst_file = os.path.join(output_dirpath, filename)
        shutil.copy(src_file, dst_file)

def main():
    file_list = os.listdir(input_dirpath)
    # 获取系统的 CPU 核心数
    num_cores = multiprocessing.cpu_count()

    # 创建一个进程池
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # 使用 tqdm 提供进度条功能
        list(tqdm(executor.map(handle_file, file_list), total=len(file_list)))



#%%
import pandas as pd
import os
import hashlib
class dir_check():
    def __init__(self,dirpath):
        self.dirpath = dirpath
        self.mapping_file = os.path.join(self.dirpath, 'hash_ligand_mapping.csv')
        self.mapping_data = pd.read_csv(self.mapping_file, header=None)
        self.smiles_list = self.mapping_data.iloc[:, 1].tolist()
        self.hash_list = self.mapping_data.iloc[:, 0].tolist()
        self.filename_list = os.listdir(self.dirpath)
        self.sdf_filename_list = [x for x in self.filename_list if x[-4:] == '.sdf']
        
    def mapping_file_check(self):
        for i in range(len(self.smiles_list)):
            if hashlib.sha1(self.smiles_list[i].encode()).hexdigest() != self.hash_list[i]:
                print('error in mapping file')
        print('mapping file check completed')
        
    def dir_file_check(self):
        filename_hash_list = [x[:-4] for x in self.sdf_filename_list]
        for filename_hash in filename_hash_list:
            if filename_hash not in self.hash_list:
                filename = filename_hash+'.sdf'
                print(filename + ' not in mapping file')
                os.remove(os.path.join(self.dirpath,filename))
                print('remove ' + os.path.join(self.dirpath,filename))
#%%      

if __name__ == '__main__':
    main()
  
    check = dir_check(output_dirpath)
    check.mapping_file_check()
    check.dir_file_check()   
    
    
    
    
    
    
    
    
    
    
    
    
    