import copy
import os
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein,  LogitsConfig
import shutil
import numpy as np
import pandas as pd
import utils
import subprocess
from Bio.PDB import PDBParser, PPBuilder
from pathlib import Path
from libs import fasta_file_reader, file_reader, save_esm_embedding
#CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1



def parse_dssp_file(dssp_path: str) -> pd.DataFrame:
    rows = []
    in_data = False
    with open(dssp_path, "r") as f:
        for line in f:
            # DSSP data begins after the header line that starts with "  #"
            if line.startswith("  #  RESIDUE"):
                in_data = True
                continue
            if not in_data:
                continue
            if not line.strip():
                continue
            try:
                dssp_idx = int(line[0:5].strip()) if line[0:5].strip() else None
                resseq = int(line[5:10].strip()) if line[5:10].strip() else None
                icode = line[10:11].strip()  # insertion code (often blank)
                chain = line[11:12].strip()
                aa = line[13:14].strip()
                ss8 = line[16:17]  # can be blank
                ss8 = "C" if ss8 == " " else ss8
                acc_str = line[34:38].strip()
                acc = float(acc_str) if acc_str else None
                # Try fixed columns for geometry (DSSP usually uses these widths)
                # If your DSSP shifts columns, we'll fall back below.
                kappa_str = line[91:97].strip()
                alpha_str = line[97:103].strip()
                phi_str   = line[103:109].strip()
                psi_str   = line[109:115].strip()
                kappa = float(kappa_str) if kappa_str else None
                alpha = float(alpha_str) if alpha_str else None
                phi   = float(phi_str) if phi_str else None
                psi   = float(psi_str) if psi_str else None
                if any(v is None for v in (kappa, alpha, phi, psi)):
                    toks = line.split()
                    last5 = toks[-5:]
                    kappa = float(last5[1])
                    alpha = float(last5[2])
                    phi   = float(last5[3])
                    psi   = float(last5[4])
                if aa == "!" or aa == "":
                    continue
                rows.append({"dssp_idx": dssp_idx,"chain": chain,"resseq": resseq,"icode": icode,"aa": aa,"ss8": ss8,
                    "acc": acc,"kappa": kappa,"alpha": alpha,"phi": phi,"psi": psi})

            except Exception:
                # If a line is malformed, skip it (or print it for debugging)
                # print("Skipping line:", line.rstrip())
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["chain", "resseq", "icode"], kind="mergesort").reset_index(drop=True)
    return df

def get_dssp_data_saved(_input,_output):
    df = parse_dssp_file(_input)
    ID = os.path.basename(_input).split(".")[0]
    ss8 = df["ss8"].values
    ss8_final = utils.one_hot_encode_ss8(ss8)
    sasa = df["acc"].values
    kappa = df["kappa"].values / 360.0
    alpha = df["alpha"].values / 360.0
    phi = df["phi"].values / 360.0
    psi = df["psi"].values / 360.0
    concat = np.column_stack((sasa,ss8_final,  kappa, alpha, phi, psi))
    output = _output + str(ID) + "_dssp.npy"
    save_esm_embedding(embedding=concat, save_path=output)
    return output

def run_dssp(_pdb_path, _output_dir):
    print("Running DSSP")
    ID = os.path.basename(_pdb_path).split(".")[0]
    dssp_path = _output_dir + "{0}.dssp".format(ID)
    pdb_path = _pdb_path
    # Automatically find DSSP executable
    dssp_exe = shutil.which("dssp") or shutil.which("mkdssp")
    if dssp_exe is None:
        raise FileNotFoundError("Neither 'dssp' nor 'mkdssp' found in PATH.")
    cryst1_header = ("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")
    # Check and add CRYST1 header if missing
    with open(pdb_path, "r") as f:
        lines = f.readlines()
    if not any(line.startswith("CRYST1") for line in lines):
        lines.insert(0, cryst1_header)
        with open(pdb_path, "w") as f:
            f.writelines(lines)
    subprocess.run([dssp_exe, "-i", str(pdb_path), "-o", str(dssp_path)],check=True)
    return str(dssp_path)

def get_intrachain_file(_antigen_pdb,_output_dir):
    print("Intra chain feature generation")
    ID = os.path.basename(_antigen_pdb).split(".")[0]
    output_file = _output_dir + "{0}_dist.npz".format(ID)
    struct_file = utils.contents_to_info(utils.read_pdb(_antigen_pdb))
    struct_file = list(filter(lambda x: (x.alt_loc != 'B'), copy.deepcopy(struct_file)))
    struct_file = utils.remove_duplicate_ca(struct_file)
    struct_file = utils.remove_residues_without_CA(struct_file)
    antigen_serial_fix_a = utils.fix_res_num_atom(struct_file)
    dist = utils.get_distance_map_heavy_atom(antigen_serial_fix_a, antigen_serial_fix_a)
    save_esm_embedding(dist, output_file)
    return output_file


def pdb_to_fasta(pdb_path, _output_dir):
    pdb_path = Path(pdb_path)
    ID = os.path.basename(pdb_path).split(".")[0]
    fasta_path = _output_dir +ID+".fasta"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    ppb = PPBuilder()

    with open(fasta_path, "w") as f:
        for chain in structure[0]:
            peptides = ppb.build_peptides(chain)
            sequence = "".join(str(p.get_sequence()) for p in peptides)

            if sequence:
                f.write(f">{pdb_path.stem}_{chain.id}\n")
                f.write(f"{sequence}\n")

    return str(fasta_path)

def geneterate_ESM3_features (_model_gen,_strcuture_path,_output_dir):
    print("ESM3 embedding generation")
    structure_protein = ESMProtein.from_pdb(_strcuture_path)
    ID= os.path.basename(_strcuture_path).split(".")[0]
    structure_token_path = _output_dir + "{0}_token.npy".format(ID)
    structure_token = _model_gen.encode(structure_protein)
    save_esm_embedding(structure_token.structure[1:-1] , structure_token_path)
    structure_embedding_path = _output_dir + "{0}_structure_embed.npy".format(ID)
    out = _model_gen.logits(structure_token, LogitsConfig(sequence=True, return_embeddings=True))
    emb = out.embeddings[:, 1:-1, :]  # remove BOS and EOS
    structure_embedding = emb.squeeze(0)
    save_esm_embedding(structure_embedding, structure_embedding_path)

    return None

input_file = "/home/rajroy/result_dump/test/8UDG_A_struture.pdb"
device = "cuda"
generative_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device).eval()
output_dir="/home/rajroy/result_dump/test/8UDG/"

if os.path.isfile(input_file) :
    print("Found file :" + input_file)
else:
    print("Missing file :" + input_file)


try:
    geneterate_ESM3_features(_model_gen=generative_model, _strcuture_path=input_file, _output_dir=output_dir)
except :
    print("Something went wrong in esm3 structure generation:" + input_file)

try:
    get_intrachain_file(input_file,output_dir)
except :
    print("Something went wrong in intra chain extraction:" + input_file)
try:
    dssp_file = run_dssp(input_file,output_dir)
    get_dssp_data_saved(dssp_file,output_dir)
except :
    print("Something went wrong in dssp running" + input_file)

pdb_to_fasta(input_file,output_dir)