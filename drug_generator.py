# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:41:07 2023

@author: Sen
"""

import os
import sys
import subprocess
import hashlib
import warnings
import platform
import csv
import numpy as np
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import shutil

import time
import subprocess
import threading
import os
import signal
import psutil

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            try:
                if os.name == 'posix':  # Unix/Linux/Mac
                    self.process = subprocess.Popen(self.cmd, shell=True, stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
                else:  # Windows
                    self.process = subprocess.Popen(self.cmd, shell=True, stderr=subprocess.DEVNULL)
                self.process.communicate()
            except Exception:
                pass

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            if os.name == 'posix':  # Unix/Linux/Mac
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:  # Windows
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            thread.join()
        return self.process.returncode if self.process else None


class LigandPostprocessor:
    def __init__(self, path):
        self.hash_ligand_mapping = {}
        self.output_path = path  # Output directory for SDF files
        self.load_mapping()

    def load_mapping(self):
        mapping_file = os.path.join(output_path, 'hash_ligand_mapping.csv')
        if os.path.exists(mapping_file):
            print("Found existed mapping file, now reading ...")
            with open(mapping_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.hash_ligand_mapping[row[0]] = row[1]

    # Define a function to save the hash-ligand mapping to a file
    def save_mapping(self):
        mapping_file = os.path.join(output_path, 'hash_ligand_mapping.csv')
        with open(mapping_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for ligand_hash, ligand in self.hash_ligand_mapping.items():
                writer.writerow([ligand_hash, ligand])

    # Define a function to filter out empty SDF files
    def filter_sdf(self, hash_ligand_mapping_per_batch):
        print("Filtering sdf ...")
        ligand_hash_list = list(hash_ligand_mapping_per_batch.keys())
        mapping_per_match = hash_ligand_mapping_per_batch.copy()
        for ligand_hash in tqdm(ligand_hash_list):
            filepath = os.path.join(self.output_path, ligand_hash + '.sdf')            
            if os.path.getsize(filepath) < 2*1024:  #2kb
                try:
                    os.remove(filepath)
                    #mapping_per_match.pop(ligand_hash)
                except Exception:
                    print(filepath)
                mapping_per_match.pop(ligand_hash)    
        return mapping_per_match

    # Define a function to generate SDF files from a list of ligand SMILES using OpenBabel
    def to_sdf(self, ligand_list_per_batch):
        print("Converting to sdf ...")
        hash_ligand_mapping_per_batch = {}
        for ligand in tqdm(ligand_list_per_batch):  
            ligand_hash = hashlib.sha1(ligand.encode()).hexdigest()
            if ligand_hash not in self.hash_ligand_mapping.keys():
                filepath = os.path.join(self.output_path , ligand_hash + '.sdf')
                
                if platform.system() == "Windows":
                    cmd = "obabel -:" + ligand + " -osdf -O " + filepath + " --gen3d --forcefield mmff94"
                elif platform.system() == "Linux":
                    obabel_path = shutil.which('obabel')
                    cmd = f"{obabel_path} -:'{ligand}' -osdf -O '{filepath}' --gen3d --forcefield mmff94"
                else:pass

                try:
                    command = Command(cmd)
                    return_code = command.run(timeout=10)
                    if return_code != 0:  # Check the return value
                        #print(f"Command execution failed with return code: {return_code}")
                        continue  # Skip the current iteration if the command execution failed
                except Exception:
                    time.sleep(1)
                    
                if os.path.exists(filepath):
                    hash_ligand_mapping_per_batch[ligand_hash] = ligand  # Add the hash-ligand mapping to the dictionary
        self.hash_ligand_mapping.update(self.filter_sdf(hash_ligand_mapping_per_batch))
    
    def delete_empty_files(self):
    # 遍历指定目录及其子目录中的所有文件
        for foldername, subfolders, filenames in os.walk(self.output_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                # 如果文件大小为0，则删除该文件
                if os.path.getsize(file_path) < 2*1024:  #2kb
                    try:
                        os.remove(file_path)
                        print(f'Deleted {file_path}')
                    except Exception:
                        pass 
    
    
    def check_sdf(self):
        file_list = os.listdir(self.output_path)
        sdf_file_list = [x for x in file_list if x[-4:]=='sdf']
        for filename in sdf_file_list:
            hash_ = filename[:-4]
            if hash_ not in self.hash_ligand_mapping.keys():
                filepath = os.path.join(self.output_path,filename)
                try:
                    os.remove(filepath)
                    print('remove ' + filepath)
                except Exception:
                    pass
            else:pass    
                
               
                
    
def about():
    print("""
  _____                    _____ _____ _______ 
 |  __ \                  / ____|  __ \__   __|
 | |  | |_ __ _   _  __ _| |  __| |__) | | |   
 | |  | | '__| | | |/ _` | | |_ |  ___/  | |   
 | |__| | |  | |_| | (_| | |__| | |      | |   
 |_____/|_|   \__,_|\__, |\_____|_|      |_|   
                     __/ |                     
                    |___/                      
 A generative drug design model based on GPT2
    """)

def ifno_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to read in FASTA file
def read_fasta_file(file_path):
    with open(file_path, 'r') as f:
        sequence = []

        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                sequence.append(line)

        protein_sequence = ''.join(sequence)
    return protein_sequence


                    
if __name__ == "__main__":
    about()
    warnings.filterwarnings('ignore')
    
    if platform.system() == "Linux":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    #Sometimes, using Hugging Face may require a proxy.
    #os.environ["http_proxy"] = "http://your.proxy.server:port"
    #os.environ["https_proxy"] = "http://your.proxy.server:port"

    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pro_seq', type=str, default=None, help='Input a protein amino acid sequence. Default value is None. Only one of -p and -f should be specified.')
    parser.add_argument('-f','--fasta', type=str, default=None, help='Input a FASTA file. Default value is None. Only one of -p and -f should be specified.')
    parser.add_argument('-l','--ligand_prompt', type=str, default='', help='Input a ligand prompt. Default value is an empty string.')
    parser.add_argument('-e','--empty_input', action='store_true', default=False, help='Enable directly generate mode.')
    parser.add_argument('-n','--number',type=int, default=100, help='At least how many molecules will be generated. Default value is 100.')
    parser.add_argument('-d','--device',type=str, default='cuda', help="Hardware device to use. Default value is 'cuda'.")
    parser.add_argument('-o','--output', type=str, default='./ligand_output/', help="Output directory for generated molecules. Default value is './ligand_output/'.")
    parser.add_argument('-b','--batch_size', type=int, default=32, help="How many molecules will be generated per batch. Try to reduce this value if you have low RAM. Default value is 32.")
    parser.add_argument('--top_k', type=int, default=9, help='The number of highest probability tokens to consider for top-k sampling. Defaults to 9.')
    parser.add_argument('--top_p', type=float, default=0.9, help='The cumulative probability threshold (0.0 - 1.0) for top-p (nucleus) sampling. It defines the minimum subset of tokens to consider for random sampling. Defaults to 0.9.')

    args = parser.parse_args()
    protein_seq = args.pro_seq
    fasta_file = args.fasta
    ligand_prompt = args.ligand_prompt
    directly_gen = args.empty_input
    num_generated = args.number
    device = args.device
    output_path = args.output
    batch_generated_size = args.batch_size
    top_k = args.top_k
    top_p = args.top_p

    
    ifno_mkdirs(output_path)
    # Check if the input is either a protein amino acid sequence or a FASTA file, but not both
    if directly_gen:
        print("Now in directly generate mode.")
        prompt = "<|startoftext|><P>"
        print(prompt)
    else:
        if (not protein_seq) and (not fasta_file):
            print("Error: Input is empty.")
            sys.exit(1)
        if protein_seq and fasta_file:
            print("Error: The input should be either a protein amino acid sequence or a FASTA file, but not both.")
            sys.exit(1)
        if fasta_file:
            protein_seq = read_fasta_file(fasta_file)
        # Generate a prompt for the model
        p_prompt = "<|startoftext|><P>" + protein_seq + "<L>"
        l_prompt = "" + ligand_prompt
        prompt = p_prompt + l_prompt
        print(prompt)


    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('liyuesen/druggpt')
    model = GPT2LMHeadModel.from_pretrained("liyuesen/druggpt")


    model.eval()
    device = torch.device(device)
    model.to(device)

    # Create a LigandPostprocessor object
    ligand_post_processor = LigandPostprocessor(output_path)

    # Generate molecules
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    batch_number = 0

    directly_gen_protein_list = []
    directly_gen_ligand_list = []

    while len(ligand_post_processor.hash_ligand_mapping) < num_generated:
        generate_ligand_list = []
        batch_number += 1
        print(f"=====Batch {batch_number}=====")
        print("Generating ligand SMILES ...")
        sample_outputs = model.generate(
            generated,
            # bos_token_id=random.randint(1,30000),
            do_sample=True,
            top_k=top_k,
            max_length=1024,
            top_p=top_p,
            num_return_sequences=batch_generated_size
        )
        for sample_output in sample_outputs:
            generate_ligand = tokenizer.decode(sample_output, skip_special_tokens=True).split('<L>')[1]
            generate_ligand_list.append(generate_ligand)
            if directly_gen:
                directly_gen_protein_list.append(tokenizer.decode(sample_output, skip_special_tokens=True).split('<L>')[0])
                directly_gen_ligand_list.append(generate_ligand)
        torch.cuda.empty_cache()
        ligand_post_processor.to_sdf(generate_ligand_list)
        ligand_post_processor.delete_empty_files()
        ligand_post_processor.check_sdf()
        
    if directly_gen:
        arr = np.array([directly_gen_protein_list, directly_gen_ligand_list])
        processed_ligand_list = ligand_post_processor.hash_ligand_mapping.values()
        with open(os.path.join(output_path, 'generate_directly.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            for index in range(arr.shape[1]):
                protein, ligand = arr[0, index], arr[1, index]
                if ligand in processed_ligand_list:
                    writer.writerow([protein, ligand])

    print("Saving mapping file ...")
    ligand_post_processor.save_mapping()
    print(f"{len(ligand_post_processor.hash_ligand_mapping)} molecules successfully generated!")
