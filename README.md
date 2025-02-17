<div class="title" align=center>
    <h1>💊DrugGPT</h1>
	<div>A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins</div>
    <br/>
    <p>
        <img src="https://img.shields.io/github/license/LIYUESEN/druggpt">
    	<img src="https://img.shields.io/badge/python-3.8-blue">
	<a href="https://colab.research.google.com/drive/1DBJWuAQc1Tl-SiIk6QWcXvBAWHQ01_kw">
	<img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
        <img src="https://img.shields.io/github/stars/LIYUESEN/druggpt?style=social">
</div>

## 💥 NEWS
**2024/08/11** We're excited to announce a new feature, Ligand Energy Minimization, now available in our latest release. Additionally, explore our new tool, druggpt_min_multi.py, designed specifically for efficient energy minimization of multiple ligands.  
**2024/07/30** All wet-lab validations have been completed, confirming that DrugGPT possesses ligand optimization capabilities.  
**2024/05/16** Wet-lab experiments confirm druggpt's ability to design ligands with new scaffolds from scratch and to repurpose existing ligands. Ligand optimization remains under evaluation. Stay tuned for more updates!  
**2024/05/16** The version has been upgraded to druggpt_v1.2, featuring new atom number control capabilities. Due to compatibility issues, the webui has been removed.  
**2024/04/03** Version upgraded to druggpt_v1.1, enhancing stability and adding a webui. Future versions will feature atom number control in molecules. Stay tuned.  
**2024/03/31** After careful consideration, I plan to create new repositories named druggpt_toolbox and druggpt_train to store post-processing tool scripts and training scripts, respectively. This repository should focus primarily on the generation of drug candidate molecules.  
**2024/03/31** I've decided to create a branch named druggpt_v1.0 for the current version since it is a stable release. Subsequently, I will continue to update the code.  
**2024/01/18** This project is now under experimental evaluation to confirm its actual value in drug research. Please continue to follow us!  

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
conda create -n druggpt python=3.8
conda activate druggpt
```
### Install PyTorch and other requirements
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install datasets transformers scipy scikit-learn psutil
conda install conda-forge/label/cf202003::openbabel
```
## 🗝 How to use
### 💻 Run in command
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
- `-b` | `--batch_size`: How many molecules will be generated per batch. Try to reduce this value if you have low RAM. Default is 16.
- `-t` | `--temperature`: Adjusts the randomness of text generation; higher values produce more diverse outputs. Default value is 1.0.
- `--top_k`: The number of highest probability tokens to consider for top-k sampling. Defaults to 9.
- `--top_p`: The cumulative probability threshold (0.0 - 1.0) for top-p (nucleus) sampling. It defines the minimum subset of tokens to consider for random sampling. Defaults to 0.9.
- `--min_atoms`: Minimum number of non-H atoms allowed for generation. Defaults to None.
- `--max_atoms`: Maximum number of non-H atoms allowed for generation. Defaults to 35.
- `--no_limit`: Disable the default max atoms limit.

  > If the `-l` | `--ligand_prompt` option is used, the `--max_atoms` and `--min_atoms` parameters will be disregarded.

### 🌎 Run in Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DBJWuAQc1Tl-SiIk6QWcXvBAWHQ01_kw)
## 🔬 Example usage 
- If you want to input a protein FASTA file
    ```shell
    python drug_generator.py -f BCL2L11.fasta -n 50
    ```
- If you want to input the amino acid sequence of the protein
    ```shell
    python drug_generator.py -p MAKQPSDVSSECDREGRQLQPAERPPQLRPGAPTSLQTEPQGNPEGNHGGEGDSCPHGSPQGPLAPPASPGPFATRSPLFIFMRRSSLLSRSSSGYFSFDTDRSPAPMSCDKSTQTPSPPCQAFNHYLSAMASMRQAEPADMRPEIWIAQELRRIGDEFNAYYARRVFLNNYQAAEDHPRMVILRLLRYIVRLVWRMH -n 50
    ```
    
- If you want to provide a prompt for the ligand  
    ```shell
    python drug_generator.py -f BCL2L11.fasta -l COc1ccc(cc1)C(=O) -n 50
    ```
    
- Note: If you are running in a Linux environment, you need to enclose the ligand's prompt with single quotes ('').  
    ```shell
    python drug_generator.py -f BCL2L11.fasta -l 'COc1ccc(cc1)C(=O)' -n 50
    ```
## 📝 How to reference this work
DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins

Yuesen Li, Chengyi Gao, Xin Song, Xiangyu Wang, Yungang Xu, Suxia Han

bioRxiv 2023.06.29.543848; doi: [https://doi.org/10.1101/2023.06.29.543848](https://doi.org/10.1101/2023.06.29.543848)

[![DOI](https://img.shields.io/badge/DOI-10.1101/2023.06.29.543848-blue)](https://doi.org/10.1101/2023.06.29.543848)
## ⚖ License
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)
