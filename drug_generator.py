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
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


warnings.filterwarnings('ignore')
#Sometimes, using Hugging Face may require a proxy.
#os.environ["http_proxy"] = "http://your.proxy.server:port"
#os.environ["https_proxy"] = "http://your.proxy.server:port"


# Set up command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default=None, help='Input the protein amino acid sequence. Default value is None. Only one of -p and -f should be specified.')
parser.add_argument('-f', type=str, default=None, help='Input the FASTA file. Default value is None. Only one of -p and -f should be specified.')
parser.add_argument('-l', type=str, default='', help='Input the ligand prompt. Default value is an empty string.')
parser.add_argument('-n', type=int, default=100, help='Number of output molecules to generate, which is not less than the input value. Default value is 100.')
parser.add_argument('-d', type=str, default='cuda', help="Hardware device to use. Default value is 'cuda'.")
parser.add_argument('-o', type=str, default='./ligand_output/', help="Output directory for generated molecules. Default value is './ligand_output/'.")
parser.add_argument('-b', type=int, default=64, help="Total number of generated molecules per batch. Try to reduce this value if you have a low RAM. Default value is 64.")


args = parser.parse_args()

protein_seq = args.p
fasta_file = args.f
ligand_prompt = args.l
num_generated = args.n
device = args.d
output_path = args.o
batch_generated_size = args.b

def ifno_mkdirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname) 

ifno_mkdirs(output_path)

# Function to read in FASTA file
def read_fasta_file(file_path):
    with open(file_path, 'r') as fasta_file:
        sequence = []
        
        for line in fasta_file:
            line = line.strip()
            if not line.startswith('>'):
                sequence.append(line)

        protein_sequence = ''.join(sequence)

    return protein_sequence

# Check if the input is either a protein amino acid sequence or a FASTA file, but not both
if (not protein_seq) and (not fasta_file):
    print("Error: Input is empty.")
    sys.exit(1)
if protein_seq and fasta_file:
    print("Error: The input should be either a protein amino acid sequence or a FASTA file, but not both.")
    sys.exit(1)
if fasta_file:
    protein_seq = read_fasta_file(fasta_file)

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('liyuesen/druggpt')
model = GPT2LMHeadModel.from_pretrained("liyuesen/druggpt")

# Generate a prompt for the model
p_prompt = "<|startoftext|><P>" + protein_seq + "<L>"
l_prompt = "" + ligand_prompt 
prompt = p_prompt + l_prompt
print(prompt)

# Move the model to the specified device
model.eval()
device = torch.device(device)
model.to(device)


# Define the post-processing class for ligands
class LigandPostprocessor:
    def __init__(self, output_path):
        self.ligand_list_tmp = []  # Temporary list to store input ligands
        self.ligand_hash_list = []  # List to store ligands after filtering
        self.output_path = output_path  # Output directory for SDF files
        self.hash_ligand_mapping = {}
        
        self.load_mapping()
    
    def load_mapping(self):
        mapping_file = os.path.join(output_path,'hash_ligand_mapping.txt')
        if os.path.exists(mapping_file):
            hash_ligand_mapping = {}
            hash_list = []
        
            with open(mapping_file, 'r') as f:
                for line in f:
                    ligand_hash, ligand = line.strip().split('\t')
                    hash_ligand_mapping[ligand_hash] = ligand
                    hash_list.append(ligand_hash)
    
                self.hash_ligand_mapping = hash_ligand_mapping
                self.ligand_hash_list = hash_list
    
    
    # Add a list of ligands to the temporary ligand list
    def add_ligand_list(self, ligand_list_input):
        self.ligand_list_tmp = ligand_list_input

    # Define a function to generate SDF files from a list of ligand SMILES using OpenBabel
    def get_sdf(self):
        print("Converting to sdf ...")
        for ligand in tqdm(self.ligand_list_tmp):
            ligand_hash = hashlib.sha1(ligand.encode()).hexdigest()
            self.hash_ligand_mapping[ligand_hash] = ligand  # Add the hash-ligand mapping to the dictionary
            filename = self.output_path + 'ligand_' + ligand_hash + '.sdf'
            cmd = "obabel -:" + ligand + " -osdf -O " + filename + " --gen3d --forcefield mmff94"
            try:
                subprocess.check_output(cmd, timeout=10, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                pass

    # Define a function to filter out empty SDF files
    def filter_sdf(self):
        filelist = os.listdir(self.output_path)
        for filename in filelist:
            filename_parts = filename.split('_')  # Split the filename string using the '_' delimiter
            ligand_hash = filename_parts[-1][:-4] # Access the last part of the split filename and remove the '.sdf' extension
            if ligand_hash not in self.ligand_hash_list:
                filepath = os.path.join(self.output_path, filename)
                with open(filepath, 'r') as f:
                    text = f.read()
                if len(text) < 2:
                    os.remove(filepath)
                else: 
                    self.ligand_hash_list.append(ligand_hash)
                    
    # Define a function to save the hash-ligand mapping to a file
    def save_mapping(self):
        mapping_file = os.path.join(output_path,'hash_ligand_mapping.txt')
        with open(mapping_file, 'w') as f:
            for ligand_hash, ligand in self.hash_ligand_mapping.items():
                if ligand_hash in self.ligand_hash_list:
                    f.write(f"{ligand_hash}\t{ligand}\n")

# Create a LigandPostprocessor object
ligand_post_processor = LigandPostprocessor(output_path)

# Generate molecules
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)


while len(os.listdir(output_path))<num_generated:
    generate_ligand_list = []
    print("generating ligand SMILES ...")
    sample_outputs = model.generate(
                                    generated, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=5, 
                                    max_length = 1024,
                                    top_p=0.6, 
                                    num_return_sequences=batch_generated_size
                                    )

    for sample_output in sample_outputs:
        generate_ligand_list.append(tokenizer.decode(sample_output, skip_special_tokens=True).split('<L>')[1])
    torch.cuda.empty_cache()
      
    ligand_post_processor.add_ligand_list(generate_ligand_list)
    ligand_post_processor.get_sdf()
    ligand_post_processor.filter_sdf()
    ligand_post_processor.save_mapping()
    



