<div class="title" align=center>
    <h1>💊DrugGPT</h1>
	<div>A generative drug design model based on GPT2</div>
    <br/>
    <p>
        <img src="https://img.shields.io/badge/license-Artistic%20License%202.0-green">
    	<img src="https://img.shields.io/badge/python-3.7-blue">
        <img src="https://img.shields.io/github/stars/LIYUESEN/druggpt?style=social">
        
</div>

## 🚩 Introduction
DrugGPT is a generative pharmaceutical strategy based on GPT structure, which aims to bring innovation to drug design by using natural language processing technique. 

This project applies the GPT model to the exploration of chemical space to discover new molecules with potential binding abilities for specific proteins. 

DrugGPT provides a fast and efficient method for the generation of drug candidate molecules by training on up to 1.8 million protein-ligand binding data.
## 📥 Deployment
### Clone
```shell
git clone https://github.com/LIYUESEN/druggpt.git
cd druggpt
```
> Or you can just click *Code>Download ZIP* to download this repo.
### Create virtual environment
```shell
conda create -n druggpt python=3.7
conda activate druggpt
```
### Download python dependencies
```shell
pip install datasets transformers scipy scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
conda install -c openbabel openbabel
```
## 🗝 How to use
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
- `-b` | `--batch_size`: How many molecules will be generated per batch. Try to reduce this value if you have low RAM. Default is 64.
## 📃 Example usage 
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
    
## ⚖ License
[Artistic License 2.0](https://opensource.org/license/artistic-license-2-0-php/)
