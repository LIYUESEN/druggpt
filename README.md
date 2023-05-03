<div class="title" align=center>
    <h1>DrugGPT</h1>
	<div>A generative drug design model based on GPT2</div>
    <br/>
    <p>
        <img src="https://img.shields.io/badge/license-Artistic%20License%202.0-green">
    	<img src="https://img.shields.io/badge/python-3.7-blue">
        <img src="https://img.shields.io/github/stars/LIYUESEN/druggpt?style=social">
        
</div>


## Deployment
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
## Example usage
Run the script with the desired arguments, such as the protein sequence, ligand prompt, number of molecules to generate, and output directory.  
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
## License
[Artistic License 2.0](https://opensource.org/license/artistic-license-2-0-php/)
