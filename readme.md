# CARTnet: Contact-Aware Deep Learning for B-cell Epitope Prediction

CARTnet is a deep learning framework for residue-level conformational B-cell epitope prediction. It integrates protein structural embeddings, multimodal residue-level features, and intra-chain contact information through contact-aware attention blocks. The framework supports both template-free prediction and template-guided prediction using known epitope residues.

## Model Architecture

![CARTnet Architecture](features_generation/data_visualization/Network_Diagram.png)

*Figure 1. Overview of the CARTnet framework for residue-level B-cell epitope prediction.*

## Installation

CARTnet was developed using Python 3.10 and PyTorch with CUDA support. All required Python packages are listed in `requirements.txt`.

**1. Clone the repository**

```bash
git clone <repository-url>
cd CARTnet
```

**2. Create and activate a Python environment**

```bash
conda create -n cartnet python=3.10 -y
conda activate cartnet
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

The provided requirements reflect the development environment, including CUDA 12.6-specific PyTorch packages. A compatible GPU environment is recommended.

## Usage

CARTnet supports template-free and template-guided epitope prediction from antigen structures in PDB format.

### Template-free Prediction

Run CARTnet without providing known epitope residues:

```bash
python main.py 
    --input_struc /path/to/antigen.pdb \
    --output_dir /path/to/output/
```

For example:

```bash
python main.py \
    --input_struc /home/rajroy/result_dump/test/8UDG_A.pdb \
    --output_dir /home/rajroy/result_dump/test/new_test_final/
```

### Template-guided Prediction

CARTnet also supports prediction using known epitope residues as templates. 

For current version template pdb structure is to be provided in the specific naming format.

The template-guided command will be added once the corresponding command-line arguments are specified.

Run CARTnet without providing known epitope residues:

```bash
python main.py \
    --input_struc /path/to/antigen.pdb \
    --output_dir /path/to/output/ \
    --template /path/to/complex_{anitbody_chains}{anigen_chains}.pdb
```

For example:

```bash
python main.py \
    --input_struc /home/rajroy/original/6W1C_H.pdb \
    --output_dir /home/rajroy/test/6W1C/ \
    --template_struc /home/rajroy/test/6VYV_PLH.pdb
```
## Citation

If you use CARTnet in your research, please cite the associated publication.

*Citation information will be updated upon publication.*