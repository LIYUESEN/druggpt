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
import csv
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


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
        for ligand_hash in tqdm(ligand_hash_list):
            filepath = os.path.join(self.output_path, ligand_hash + '.sdf')
            with open(filepath, 'r') as f:
                text = f.read()
            if len(text) < 2:
                os.remove(filepath)
                hash_ligand_mapping_per_batch.pop(ligand_hash)
        return hash_ligand_mapping_per_batch

    # Define a function to generate SDF files from a list of ligand SMILES using OpenBabel
    def to_sdf(self, ligand_list_per_batch):
        print("Converting to sdf ...")
        hash_ligand_mapping_per_batch = {}
        for ligand in tqdm(ligand_list_per_batch):
            ligand_hash = hashlib.sha1(ligand.encode()).hexdigest()
            if ligand_hash not in self.hash_ligand_mapping.keys():
                filepath = self.output_path + ligand_hash + '.sdf'
                cmd = "obabel -:" + ligand + " -osdf -O " + filepath + " --gen3d --forcefield mmff94"
                try:
                    subprocess.check_output(cmd, timeout=10, stderr=subprocess.DEVNULL)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    if os.path.exists(filepath):
                        hash_ligand_mapping_per_batch[
                            ligand_hash] = ligand  # Add the hash-ligand mapping to the dictionary
        self.hash_ligand_mapping.update(self.filter_sdf(hash_ligand_mapping_per_batch))

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
    #Sometimes, using Hugging Face may require a proxy.
    #os.environ["http_proxy"] = "http://your.proxy.server:port"
    #os.environ["https_proxy"] = "http://your.proxy.server:port"

    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pro_seq', type=str, default=None, help='Input a protein amino acid sequence. Default value is None. Only one of -p and -f should be specified.')
    parser.add_argument('-f','--fasta', type=str, default=None, help='Input a FASTA file. Default value is None. Only one of -p and -f should be specified.')
    parser.add_argument('-l','--ligand_prompt', type=str, default='', help='Input a ligand prompt. Default value is an empty string.')
    parser.add_argument('-n','--number',type=int, default=100, help='At least how many molecules will be generated. Default value is 100.')
    parser.add_argument('-d','--device',type=str, default='cuda', help="Hardware device to use. Default value is 'cuda'.")
    parser.add_argument('-o','--output', type=str, default='./ligand_output/', help="Output directory for generated molecules. Default value is './ligand_output/'.")
    parser.add_argument('-b','--batch_size', type=int, default=64, help="How many molecules will be generated per batch. Try to reduce this value if you have low RAM. Default value is 64.")

    args = parser.parse_args()
    protein_seq = args.p
    fasta_file = args.f
    ligand_prompt = args.l
    num_generated = args.n
    device = args.d
    output_path = args.o
    batch_generated_size = args.b
    
    ifno_mkdirs(output_path)
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

    # Create a LigandPostprocessor object
    ligand_post_processor = LigandPostprocessor(output_path)

    # Generate molecules
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    batch_number = 0


    while len(os.listdir(output_path))<num_generated:
        generate_ligand_list = []
        batch_number += 1
        print(f"Batch {batch_number}")
        print("Generating ligand SMILES ...")
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
        ligand_post_processor.to_sdf(generate_ligand_list)
    print("Saving mapping file ...")
    ligand_post_processor.save_mapping()
    print(f"{len(ligand_post_processor.hash_ligand_mapping)} molecules successfully generated!")



