# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:41:07 2023

@author: Sen
"""

import os
import subprocess
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
parser.add_argument('-n', type=int, default=100, help='Number of output molecules to generate. Default value is 100.')
parser.add_argument('-d', type=str, default='cuda', help="Hardware device to use. Default value is 'cuda'.")
parser.add_argument('-o', type=str, default='./ligand_output/', help="Output directory for generated molecules. Default value is './ligand_output/'.")

args = parser.parse_args()

protein_seq = args.p
fasta_file = args.f
ligand_prompt = args.l
num_generated = args.n
device = args.d
output_path = args.o


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
if (protein_seq is not None) != (fasta_file is not None):
    if fasta_file is not None:
        protein_seq = read_fasta_file(fasta_file)
    else:
        protein_seq = protein_seq
else:
    print("The input should be either a protein amino acid sequence or a FASTA file, but not both.")

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



#Define post-processing function
#Define function to generate SDF files from a list of ligand SMILES using OpenBabel
def get_sdf(ligand_list,output_path):
    for ligand in tqdm(ligand_list):
        filename = output_path + 'ligand_' + ligand +'.sdf'
        cmd = "obabel -:" + ligand + " -osdf -O " + filename + " --gen3d --forcefield mmff94"# --conformer --nconf 1 --score rmsd
        #subprocess.check_call(cmd, shell=True)
        try:
            output = subprocess.check_output(cmd, timeout=10)
        except subprocess.TimeoutExpired:
            pass
#Define function to filter out empty SDF files
def filter_sdf(output_path):
    filelist = os.listdir(output_path)
    for filename in filelist:
        filepath = os.path.join(output_path,filename)
        with open(filepath,'r') as f:
            text = f.read()
        if len(text)<2:
            os.remove(filepath)




# Generate molecules
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)


for i in range(100):
    ligand_list = []
    sample_outputs = model.generate(
                                    generated, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=5, 
                                    max_length = 1024,
                                    top_p=0.6, 
                                    num_return_sequences=64
                                    )

    for i, sample_output in enumerate(sample_outputs):
        ligand_list.append(tokenizer.decode(sample_output, skip_special_tokens=True).split('<L>')[1])
    torch.cuda.empty_cache()
      
    get_sdf(ligand_list,output_path)
    filter_sdf(output_path)

    if len(os.listdir(output_path))>num_generated:
        break
    else:pass



