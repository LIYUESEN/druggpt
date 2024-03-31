<div class="title" align=center>
    <h1>💊DrugGPT</h1>
	<div>A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins</div>
    <br/>
    <p>
        <img src="https://img.shields.io/github/license/LIYUESEN/druggpt">
    	<img src="https://img.shields.io/badge/python-3.7-blue">
	<a href="https://colab.research.google.com/drive/1x7w6LcgkB4kxDDVny4SRVIvvjkUe8vbE#scrollTo=2h2QAp7EqgyY">
	<img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
        <img src="https://img.shields.io/github/stars/LIYUESEN/druggpt?style=social">
</div>

## 💥 NEWS
**2024/01/18** This project is now under experimental evaluation to confirm its actual value in drug research. Please continue to follow us!  
**2024/03/31** I've decided to create a branch named druggpt_v1.0 for the current version since it is a stable release. Subsequently, I will continue to update the code.  
**2024/03/31** After careful consideration, I plan to create new repositories named druggpt_toolbox and druggpt_train to store post-processing tool scripts and training scripts, respectively. This repository should focus primarily on the generation of drug candidate molecules.

## 🚩 Introduction
DrugGPT presents a ligand design strategy based on the autoregressive model, GPT, focusing on chemical space exploration and the discovery of ligands for specific proteins. Deep learning language models have shown significant potential in various domains including protein design and 
biomedical text analysis, providing strong support for the proposition of DrugGPT. 

In this study, we employ the DrugGPT model to learn a substantial amount of protein-ligand binding data, aiming to discover novel molecules that can bind with specific proteins. This strategy not only significantly improves the efficiency of ligand design but also offers a swift and effective avenue for the drug development process, bringing new possibilities to the pharmaceutical domain
## 📥 Deployment
### Clone
```shell
git clone https://github.com/LIYUESEN/druggpt.git
cd druggpt
```
> Or you can just click *Code>Download ZIP* to download this repo.
### Create Python virtual environment
```shell
conda create -n druggpt python=3.7
conda activate druggpt
```
### Install PyTorch and other requirements
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install datasets transformers scipy scikit-learn
conda install -c openbabel openbabel
```
## 🗝 How to use
### 💻 Run locally
Use [drug_generator.py](https://github.com/LIYUESEN/druggpt/blob/main/drug_generator.py)

Required parameters:
- `-p` | `--pro_seq`: Input a protein amino acid sequence.
- `-f` | `--fasta`: Input a FASTA file.

  > Only one of -p and -f should be specified.
- `-l` | `--ligand_prompt`: Input a ligand prompt.
- `-e` | `--empty_input`: Enable directly generate mode.
- `-n` | `--number`: At least how many molecules will be generated.
- `-d` | `--device`: Hardware device to use. Default is 'cuda'.
- `-o` | `--output`: Output directory for generated molecules. Default is './ligand_output/'.
- `-b` | `--batch_size`: How many molecules will be generated per batch. Try to reduce this value if you have low RAM. Default is 32.
- `--top_k`: The number of highest probability tokens to consider for top-k sampling. Defaults to 9.
- `--top_p`: The cumulative probability threshold (0.0 - 1.0) for top-p (nucleus) sampling. It defines the minimum subset of tokens to consider for random sampling. Defaults to 0.9.
### 🌎 Run in Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x7w6LcgkB4kxDDVny4SRVIvvjkUe8vbE#scrollTo=2h2QAp7EqgyY)
## 🔬 Example usage 
- If you want to input a protein FASTA file
    ```shell
    python drug_generator.py -f bcl2.fasta -n 50
    ```
- If you want to input the amino acid sequence of the protein
    ```shell
    python drug_generator.py -p MAKQPSDVSSECDREGRQLQPAERPPQLRPGAPTSLQTEPQGNPEGNHGGEGDSCPHGSPQGPLAPPASPGPFATRSPLFIFMRRSSLLSRSSSGYFSFDTDRSPAPMSCDKSTQTPSPPCQAFNHYLSAMASMRQAEPADMRPEIWIAQELRRIGDEFNAYYARRVFLNNYQAAEDHPRMVILRLLRYIVRLVWRMH -n 50
    ```
    
- If you want to provide a prompt for the ligand  
    ```shell
    python drug_generator.py -f bcl2.fasta -l COc1ccc(cc1)C(=O) -n 50
    ```
    
- Note: If you are running in a Linux environment, you need to enclose the ligand's prompt with single quotes ('').  
    ```shell
    python drug_generator.py -f bcl2.fasta -l 'COc1ccc(cc1)C(=O)' -n 50
    ```
## 📝 How to reference this work
DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins

Yuesen Li, Chengyi Gao, Xin Song, Xiangyu Wang, Yungang Xu, Suxia Han

bioRxiv 2023.06.29.543848; doi: [https://doi.org/10.1101/2023.06.29.543848](https://doi.org/10.1101/2023.06.29.543848)

[![DOI](https://img.shields.io/badge/DOI-10.1101/2023.06.29.543848-blue)](https://doi.org/10.1101/2023.06.29.543848)
## ⚖ License
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)
