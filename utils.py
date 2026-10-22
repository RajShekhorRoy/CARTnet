import copy
import csv
import pickle
import pandas as pd
import os
from Bio.Align import PairwiseAligner
from libs import check_single_exists, text_file_reader, save_esm_embedding
import re

INTRA_CHAIN_CONTACT_TRHESHOLD = 8
header_re = re.compile(r"^>Cluster\s+(\d+)\s*$")
AVOID_LIST = ["7UYM_K"]
propenisity_dict= {"A":1.064,"C":1.412,"D":0.866,"E":0.851,"F":1.091,"G":0.874,"H":1.105,
                   "I":1.152,"K":0.93,"L":1.25,"M":0.826,"N":0.776,"P":1.064,"Q":1.105,
                   "R":0.873,"S":1.012,"T":0.909,"V":1.383,"W":0.893,"Y":1.161}


hydrophobicity = {
    'I': 4.5,
    'V': 4.2,
    'L': 3.8,
    'F': 2.8,
    'C': 2.5,
    'M': 1.9,
    'A': 1.8,
    'W': -0.9,
    'G': -0.4,
    'T': -0.7,
    'S': -0.8,
    'Y': -1.3,
    'P': -1.6,
    'H': -3.2,
    'N': -3.5,
    'D': -3.5,
    'Q': -3.5,
    'E': -3.5,
    'K': -3.9,
    'R': -4.5
}
charge = {
    'I': 0,
    'V': 0,
    'L': 0,
    'F': 0,
    'C': 0,
    'M': 0,
    'A': 0,
    'W': 0,
    'G': 0,
    'T': 0,
    'S': 0,
    'Y': 0,
    'P': 0,
    'H': 0.1,
    'N': 0,
    'D': -1,
    'Q': 0,
    'E': -1,
    'K': 1,
    'R': 1
}

def count_uppercase(s):
    return sum(1 for c in s if c.isupper())
def get_fasta_format (values, label_string):
    fasta_format = ">{0}\n{1}".format(values, label_string)
    return fasta_format

def pad_to_appropitate_L(tensor, pad_value=0.0,MAX_LENGTH=1400):
    L, D = tensor.shape  # L: current length, D: 2593
    if L >= MAX_LENGTH:
        return tensor[:MAX_LENGTH]  # Truncate if longer
    pad_rows = MAX_LENGTH - L
    padding = torch.full((pad_rows, D), pad_value, dtype=tensor.dtype)
    padded_tensor = torch.cat([tensor, padding], dim=0)
    return padded_tensor


def pad_to_square(tensor, pad_value=0.0, MAX_LENGTH=1400):

    # Truncate if needed
    tensor = tensor[:MAX_LENGTH, :MAX_LENGTH]
    L1, L2 = tensor.shape

    # Compute padding
    pad_rows = MAX_LENGTH - L1
    pad_cols = MAX_LENGTH - L2

    # Pad rows
    if pad_rows > 0:
        row_pad = torch.full((pad_rows, L2), pad_value, dtype=tensor.dtype)
        tensor = torch.cat([tensor, row_pad], dim=0)

    # Pad columns
    if pad_cols > 0:
        col_pad = torch.full((MAX_LENGTH, pad_cols), pad_value, dtype=tensor.dtype)
        tensor = torch.cat([tensor, col_pad], dim=1)

    return tensor



def first_x_ones(x, l):
    return [1]*x + [0]*(l-x)

def load_embedding(load_path ):
    emb = np.load(load_path )
    return emb

def filter_files(_fasta_dir, _esm_dir, _ss8_dir, _seq_list, _sasa_dir,_token_dir,_label_dir):
    _esm_files = []
    _ss8_files = []
    _fasta_files = []
    _labels = []
    _sasa_files = []
    _token_files = []
    _names = []

    for val in _seq_list:
        if val not in AVOID_LIST:
            # esm_file = _esm_dir + str(val) + "_sequence.npy"
            esm_file = _esm_dir + str(val) + "_esm2.npz.npy"
            ss8_file = _ss8_dir + str(val) + "_ss8.npy"

            fasta_file = _fasta_dir + val + ".fasta"
            sasa_path = _sasa_dir+str(val) + "_sasa.npy"
            token_path = _token_dir +str(val) + "_token.npy"
            label_path = _label_dir +str(val) + ".txt"
            # print(fasta_file,esm_path,ss8_file,label_value)

            if not (check_single_exists(esm_file) or check_single_exists(ss8_file) or check_single_exists(fasta_file)or
                    check_single_exists(label_path) or check_single_exists(sasa_path) or  check_single_exists(token_path) or check_single_exists(label_path)):
                print(esm_file, ss8_file, fasta_file,label_path, sasa_path, token_path, label_path)
            else:

                _names.append(os.path.basename(val).split(".")[0])
                _esm_files.append(esm_file)
                _ss8_files.append(ss8_file)
                _fasta_files.append(fasta_file)
                _labels.append(label_path)
                _sasa_files.append(sasa_path)
                _token_files.append(token_path)


    return _fasta_files, _esm_files, _ss8_files, _labels,_names,_sasa_files,_token_files

def get_seq_propensity(_seq):
    embed=[]
    for values in _seq:
        embed.append(propenisity_dict[values])

    return embed

def generate_label_array(value,seq_len, max_length=600):
    def_array = np.zeros([max_length])
    if value == 0:
        return def_array
    else:
        for i in range(seq_len):
            def_array[i] = value

        return def_array

def ss8_feature_reader(_file):
    df = pd.read_csv(_file)
    df[" psi"]=df[" psi"] / 180.0
    df[" phi"] = df[" phi"] / 180.0
    # df = pd.read_csv(_file, usecols=[" rsa"," p[q8_G]"	," p[q8_H]"," p[q8_I]"	," p[q8_B]"	," p[q8_E]"," p[q8_S]"," p[q8_T]"," p[q8_C]"," phi"	," psi",	" disorder"])
    selected_cols = df[
        [" rsa", " p[q8_G]", " p[q8_H]", " p[q8_I]", " p[q8_B]", " p[q8_E]", " p[q8_S]", " p[q8_T]", " p[q8_C]", " phi",
         " psi", " disorder"]]
    final_out = selected_cols.to_numpy()
    return final_out


def save_checkpoint(_model=None, optimizer=None, _epoch=0,HISTORY_FILE="/home/rajroy/venus_seq_features/training/"):
    state = {
        'epoch': _epoch,
        'model': _model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, HISTORY_FILE + "/weighths_" + str(_epoch))
    print(HISTORY_FILE + "/weighths_" + str(_epoch))
    return  HISTORY_FILE + "/weighths_" + str(_epoch)

def get_esm_feat(_esm_file):
    with open(_esm_file, "rb") as f:
        esm_feat = pickle.load(f)
    # print(_esm_file.shape)
    return esm_feat.cpu().detach().numpy()

def read_lables_from_one2zero_indexed(path):
    with open(path, "r") as f:
        content = f.read().strip()

    # split by comma, drop empty entries, convert to int
    arr = [int(x)-1 for x in content.split(",") if x.strip()]

    return arr

def presence_array(arr, length):
    result = [0] * length
    for x in arr:
        result[x ] = 1
    return result
def sasa_to_rsa_using_95pct(sasa_array):
    sasa = np.array(sasa_array, dtype=float)

    # 95th percentile value
    max95 = np.percentile(sasa, 95)

    # clip higher values down to 95th percentile
    sasa_clipped = np.clip(sasa, 0, max95)

    # normalize
    rsa = sasa_clipped / max95

    return rsa

def one_hot_encode_fasta(sequence, max_len=None):
    # 20 standard amino acids + one for unknowns (X)
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    NUM_AMINO_ACIDS = len(AMINO_ACIDS) + 1  # 20 + 1 for unknowns
    """One-hot encode a protein sequence. Pads with zeros if max_len is provided."""
    seq_encoded = np.zeros((max_len or len(sequence), NUM_AMINO_ACIDS), dtype=np.float32)

    for i, aa in enumerate(sequence[:max_len]):
        idx = AA_TO_INDEX.get(aa.upper(), NUM_AMINO_ACIDS - 1)  # Unknown -> last index
        seq_encoded[i, idx] = 1.0

    return seq_encoded




def one_hot_encode_ss8(sequence, max_len=None):
    SS8_CHARS = "HBEGITSC"  # 8 DSSP states
    SS8_TO_INDEX = {c: i for i, c in enumerate(SS8_CHARS)}

    length = max_len or len(sequence)
    arr = np.zeros((length, len(SS8_CHARS)), dtype=np.float32)

    for i, c in enumerate(sequence[:length]):
        idx = SS8_TO_INDEX.get(c.upper())
        if idx is not None:  # ignore anything not in SS8
            arr[i, idx] = 1.0
    return arr
def get_esm_embedding(_esm_file):
    _esm_feat = load_embedding(_esm_file)  # 1152
    _esm_feature_pad = pad_to_appropitate_L(torch.from_numpy(_esm_feat))
    return _esm_feature_pad
import numpy as np

def angle_to_sincos(angle_array_deg):

    # degrees -> radians
    angle_rad = np.deg2rad(angle_array_deg)

    sin_feat = np.sin(angle_rad)
    cos_feat = np.cos(angle_rad)

    # concatenate on last dimension
    out = np.concatenate([sin_feat, cos_feat], axis=-1)

    return out
def get_sasa_embedding(_esm_file):
    _dssp_val = load_embedding(_esm_file)  # 1152
    # _sasa_val = np.expand_dims(sasa_to_rsa_using_95pct(_dssp_val[:, 0]),axis=-1)
    _angles= angle_to_sincos(_dssp_val[:,9:]*360)
    # _final_arrr = np.concatenate([_dssp_val[:,1:9],_sasa_val, _angles], axis=-1)
    _final_arrr = np.concatenate([_dssp_val[:,1:9], _angles], axis=-1)
    _sasa_feature_pad = pad_to_appropitate_L(torch.from_numpy(_final_arrr))

    return _sasa_feature_pad

def get_labels (_file,_length):
    _labels = read_lables_from_one2zero_indexed(_file)
    _labels_pad = torch.tensor(pad_to_appropitate_L(torch.tensor(presence_array(_labels, _length),
                                                                dtype=torch.long).unsqueeze(dim=1)), dtype=torch.long)
    return _labels_pad

def get_ss8_feature(_ss8_file):
    _ss8_data = load_embedding(_ss8_file)  #8
    _ss8_feature = one_hot_encode_ss8(_ss8_data)
    _ss8_feature_pad = pad_to_appropitate_L(torch.from_numpy(_ss8_feature))
    return  _ss8_feature_pad

def get_sasa_feature(_sasa_file):
    _sasa_feature_initial = load_embedding(_sasa_file)
    # _sasa_feature = sasa_to_rsa_using_95pct(_sasa_feature_initial)
    _sasa_feature_pad = pad_to_appropitate_L(torch.from_numpy(_sasa_feature_initial).unsqueeze(dim=1))
    return _sasa_feature_pad

def seq_propenisty_feature(_fasta_file):
    _seq_prop = np.expand_dims(np.array(get_seq_propensity(text_file_reader(_fasta_file)[1])), axis=1)
    _seq_prop_pad = pad_to_appropitate_L(torch.from_numpy(_seq_prop))
    return _seq_prop_pad

def token_embedding_feature(_token_file):
    _token_feat = load_embedding(_token_file)  # 1
    _token_feat_pad = pad_to_appropitate_L(
        torch.tensor(torch.from_numpy(_token_feat), dtype=torch.float32).unsqueeze(dim=1))
    return _token_feat_pad

def get_sequence_identity_features (_fasta_file):
    _seq_identity_feature = one_hot_encode_fasta(text_file_reader(_fasta_file)[1]) #21
    _seq_identity_feature_pad = pad_to_appropitate_L(torch.from_numpy(_seq_identity_feature))
    return _seq_identity_feature_pad

def uppercase_indexes(s):
    return [i for i, ch in enumerate(s) if ch.isupper()]

def get_20_value(_arr):
    sorted_arr = sorted(_arr)
    cutoff_index = int(0.8 * len(sorted_arr))
    threshold = sorted_arr[cutoff_index]
    return threshold

def get_top_n_value(_arr,_n):
    sorted_arr = sorted(_arr)
    cutoff_index = int(_n* len(sorted_arr))
    threshold = sorted_arr[cutoff_index]
    return threshold

def top_n_percent_binary(arr,_n):
    threshold = get_top_n_value(arr,_n)
    return [1 if x >= threshold else 0 for x in arr]


def get_20_value_tensor(arr):
    sorted_arr, _ = torch.sort(arr, dim=1)

    cutoff_index = int(0.8 * arr.shape[1])

    threshold = sorted_arr[:, cutoff_index]

    return threshold

def top_20_percent_binary(arr):

    threshold = get_20_value(arr)

    return [1 if x >= threshold else 0 for x in arr]
def get_labels_BP3 (_seq):
    data_label = uppercase_indexes(_seq)
    L= len(_seq)
    _array = np.zeros(L)
    _array[data_label] = 1
    _labels_pad = torch.tensor(pad_to_appropitate_L(torch.tensor(_array, dtype=torch.long).unsqueeze(dim=1)), dtype=torch.long)
    return _labels_pad

def get_processed_data(_esm_file, _ss8_file, _fasta_file, _sasa_file, _token_file,_labels_file):
    name = os.path.basename(_fasta_file)
    if (check_single_exists(_esm_file) or check_single_exists(_ss8_file) or check_single_exists(_fasta_file) or
            check_single_exists(_sasa_file) or check_single_exists(_token_file) or check_single_exists(_labels_file)) :

        print(_esm_file, _ss8_file, _fasta_file, _sasa_file, _token_file,_labels_file)

        try:
            seq_identity_feature_pad = get_sequence_identity_features(_fasta_file)  # 21
        except:
            print("Fasta is the issue", os.path.basename(_fasta_file))

        try:
            ss8_feature_pad = get_ss8_feature(_ss8_file)  # 8
        except:
            print("SS8 is the issue", os.path.basename(_ss8_file))

        try:
            esm_feature_pad = get_esm_embedding(_esm_file)  # 1152
        except:
            print("esm_feat is the issue", os.path.basename(_esm_file))

        try:
            sasa_feature_pad = get_sasa_feature(_sasa_file)  # 1     #needs rsa
        except:
            print("SASA is the issue", os.path.basename(_sasa_file))

        try:
            seq_prop_pad = seq_propenisty_feature(_fasta_file)  # 1
        except:
            print("seq_prop is the issue", os.path.basename(_fasta_file))

        try:
            token_feat_pad = token_embedding_feature(_token_file)  # 1
        except:
            print("token is the issue", os.path.basename(_token_file))

        try:
            function_feat = torch.tensor(
                torch.cat([seq_identity_feature_pad, ss8_feature_pad, sasa_feature_pad, seq_prop_pad], axis=1),
                dtype=torch.float32)
        except:
            print("concat error")

        sequence_length = len(text_file_reader(_fasta_file)[1])
        try:
            labels_pad =get_labels( _labels_file,sequence_length)
        except:
            labels_pad = []
            print ("no labels provided")


            # esm, strcutrure, function
        return {'feat': function_feat.permute(0, 1).unsqueeze(0), 'esm': esm_feature_pad.permute(1, 0).unsqueeze(0),
                'structure_token': token_feat_pad.permute(0, 1).unsqueeze(0),'ground_truth': labels_pad.unsqueeze(0),'name':name,"length":sequence_length}

def get_top_results(_pred,_target,_k_values,_lengths):
    array_result = []
    for _y,_y_hat,_k,_l in zip(_target,_pred,_k_values,_lengths):
        one_pred = np.asarray(_y_hat).reshape(-1)[0:_l]
        one_target = np.asarray(_y).reshape(-1)[0:_l]
        # Get indices of top-K predictions (descending)
        topk_idx = np.argsort(one_pred)[- _k:][::-1]

        # Count positives in top-K
        tp_at_k = one_target[topk_idx].sum()

        precision_at_k = tp_at_k / _k
        array_result.append(precision_at_k)
    return np.mean(array_result)


def build_accession_dict(csv_file):
    accession_dict = {}

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue

            accession = row[0]          # 1st column
            third_col_value = float(row[-1])    # 3rd column
            #for bepipred2 -1
            if accession not in accession_dict:
                accession_dict[accession] = []

            accession_dict[accession].append(third_col_value)

    return accession_dict

#important bepipred files to numpy
def convert_raw_bepi_output_files (_input="/home/rajroy/bepipred2_benchmark/testset/summary_6AAAC06900218582F31D5487.csv",
                                   output="/home/rajroy/bepipred2_benchmark/testset/np_pred/"):
    values = build_accession_dict(_input)
    for val in values:
        pred_arr = values[val]
        save_esm_embedding(pred_arr, output+val+"_predicted_labels.npy")
    return

# convert_raw_bepi_output_files()



# def cdhit_parse_clstr(path):
#     clusters = defaultdict(list)
#     rep = None
#
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#
#             if line.startswith(">Cluster"):
#                 rep = None
#                 continue
#
#             # extract sequence ID like 2NYY_A from >2NYY_A...
#             m = re.search(r">([^.\s,]+)\.\.\.", line)
#
#             if not m:
#                 continue
#
#             seq_id = m.group(1)
#
#             if line.endswith("*"):
#                 rep = seq_id
#                 clusters[rep]  # ensure key exists
#             else:
#                 clusters[rep].append(seq_id)
#
#     return dict(clusters)

def caps_at_indexes(s, idxs):
    idxs = set(idxs)  # O(1) lookup
    return "".join(
        c.upper() if i in idxs else c
        for i, c in enumerate(s)
    )


def merge_case(strings):
    if not strings:
        return ""

    length = len(strings[0])
    assert all(len(s) == length for s in strings), "All strings must be same length"

    result = []

    for i in range(length):
        chars = [s[i] for s in strings]
        base = chars[0].lower()

        if any(c.isupper() for c in chars):
            result.append(base.upper())
        else:
            result.append(base)

    return "".join(result)
#not good stuggles with addin tail
def get_aligned_sequences_old(seq1, seq2, mode="global"):
    aligner = PairwiseAligner()
    aligner.mode = mode

    alignment = aligner.align(seq1, seq2)[0]

    aligned1 = []
    aligned2 = []

    pos1, pos2 = 0, 0

    for (start1, end1), (start2, end2) in zip(alignment.aligned[0], alignment.aligned[1]):
        # Add gaps before next block
        while pos1 < start1:
            aligned1.append(seq1[pos1])
            aligned2.append("-")
            pos1 += 1

        while pos2 < start2:
            aligned1.append("-")
            aligned2.append(seq2[pos2])
            pos2 += 1

        # Add matched block
        while pos1 < end1 and pos2 < end2:
            aligned1.append(seq1[pos1])
            aligned2.append(seq2[pos2])
            pos1 += 1
            pos2 += 1

    return "".join(aligned1), "".join(aligned2)

def get_aligned_sequences(seq1, seq2, mode="global"):
    aligner = PairwiseAligner()
    aligner.mode = mode

    alignment = aligner.align(seq1, seq2)[0]

    aligned1 = []
    aligned2 = []
    pos1, pos2 = 0, 0

    for (start1, end1), (start2, end2) in zip(alignment.aligned[0], alignment.aligned[1]):
        # gaps before next aligned block
        while pos1 < start1:
            aligned1.append(seq1[pos1])
            aligned2.append("-")
            pos1 += 1

        while pos2 < start2:
            aligned1.append("-")
            aligned2.append(seq2[pos2])
            pos2 += 1

        # aligned block
        while pos1 < end1 and pos2 < end2:
            aligned1.append(seq1[pos1])
            aligned2.append(seq2[pos2])
            pos1 += 1
            pos2 += 1

    # IMPORTANT: append any trailing residues after the last block
    while pos1 < len(seq1):
        aligned1.append(seq1[pos1])
        aligned2.append("-")
        pos1 += 1

    while pos2 < len(seq2):
        aligned1.append("-")
        aligned2.append(seq2[pos2])
        pos2 += 1

    return "".join(aligned1), "".join(aligned2)

import re



def cdhit_rep_to_members(path: str):
    ID_RE = re.compile(r">([^.\s,]+)\.\.\.")
    rep_to_members = {}

    members = []
    rep = None
    in_cluster = False

    def flush():
        nonlocal members, rep
        if rep is not None:
            rep_to_members[rep] = [m for m in members if m != rep]
        members = []
        rep = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">Cluster"):
                if in_cluster:
                    flush()
                in_cluster = True
                continue

            if not in_cluster:
                continue

            m = ID_RE.search(line)
            if not m:
                continue

            seq_id = m.group(1)
            members.append(seq_id)

            if line.endswith("*"):
                rep = seq_id

    if in_cluster:
        flush()

    return rep_to_members

def nglyco_mask(
    seq: str,
    mark="triplet",          # "triplet" = mark N-X-S/T positions; "N_only" = mark only the N position
    ignore_case=True,
    dtype=np.int8
) -> np.ndarray:
    """
    Returns an array of length len(seq) with 1 where an N-glycosylation motif occurs
    (N[^P][ST]) and 0 elsewhere.
    """
    s = seq.upper() if ignore_case else seq
    mask = np.zeros(len(s), dtype=dtype)

    # Find overlapping matches of N[^P][ST]
    for m in re.finditer(r'(?=(N[^P][ST]))', s):
        start = m.start()
        if mark == "N_only":
            mask[start] = 1
        elif mark == "triplet":
            mask[start:start+3] = 1
        else:
            raise ValueError("mark must be 'triplet' or 'N_only'")

    return mask
def split_line_to_tuple(line):
    a_pdb_line = pdb_lines()

    a_pdb_line.atom = line[0:6].strip()
    a_pdb_line.serial = line[6:12].strip()
    a_pdb_line.atom_name = line[12:16].strip()
    a_pdb_line.alt_loc = line[16].strip()
    a_pdb_line.res_name = line[17:20].strip()
    a_pdb_line.chain = line[20:22].strip()
    a_pdb_line.res_num = line[22:26].strip()
    a_pdb_line.orig_res_num = line[22:26].strip()
    a_pdb_line.icode = line[26:30].strip()
    a_pdb_line.x = line[30:38].strip()
    a_pdb_line.y = line[38:46].strip()
    a_pdb_line.z = line[46:54].strip()
    a_pdb_line.occupancy = line[54:60].strip()
    # a_pdb_line.temp_fact = line[60:76].strip()
    a_pdb_line.temp_fact = line[60:66].strip()
    a_pdb_line.element = line[76:78].strip()
    a_pdb_line.charge = line[78:80].strip()

    return a_pdb_line

def contents_to_info(contents):  # reads the ATOM line. Then splits the info into respective frames and returns the data
    split_contents = []
    for lines in contents:
        if lines.startswith("ATOM"):
            pdb_line = split_line_to_tuple(lines.strip())
            split_contents.append(pdb_line)
    return split_contents


def fix_res_num_atom(_pdb):
    ca_index = 0
    prev_tag = ""
    # _pdb_ = increase_residue_n(copy.deepcopy(copy.deepcopy(_pdb)))
    for values in _pdb:

        current_tag = values.res_name + '_' + str(values.orig_res_num) + "_" + values.icode
        # print(current_tag,prev_tag,ca_index)

        if current_tag != prev_tag:

            ca_index += 1

            list_atoms = list(filter(lambda x: (x.orig_res_num == values.orig_res_num), _pdb))
            list_atoms = list(filter(lambda x: (x.res_name == values.res_name), list_atoms))
            list_atoms = list(filter(lambda x: (x.icode == values.icode), list_atoms))
            list_atoms = list(filter(lambda x: (x.visited == 0), list_atoms))

            for list_atom in list_atoms:
                list_atom.res_num = str(ca_index)
                list_atom.visited = 1
            # prev_tag =current_tag

            prev_tag = values.res_name + '_' + str(values.orig_res_num) + "_" + values.icode
    _pdb = fix_serial(_pdb)
    return _pdb

def fix_serial(_array, _no=1):
    number = _no
    for x in _array:
        x.serial = number
        number = number + 1
    return _array

def get_distance(a_cord,b_cord):
    value = np.sqrt((float(a_cord.x) - float(b_cord.x)) ** 2 + (float(a_cord.y) - float(b_cord.y)) ** 2 + (            float(a_cord.z) - float(b_cord.z)) ** 2)
    return value
def get_distance_resdiues(__first_chain,__second_chain):
    min_dist=99999
    for value_a in __first_chain:
        for value_b in __second_chain:
            res_dist = get_distance(value_a,value_b)
            if res_dist<min_dist:
                min_dist=res_dist
    return min_dist

def remove_duplicate_ca(atoms):
    """
    Take a list of pdb_lines objects and remove extra CA atoms per residue.
    Keeps only the FIRST CA seen for each (chain, res_num, icode, res_name).
    Does NOT renumber residues.
    """
    seen_ca = set()
    new_atoms = []

    for a in atoms:
        if a.atom_name.strip() == "CA":
            key = (a.chain, str(a.res_num), a.icode or "", a.res_name)
            if key in seen_ca:
                # already have a CA for this residue → drop this one
                continue
            seen_ca.add(key)

        # keep all non-CA, and the first CA per residue
        new_atoms.append(a)

    return new_atoms
def get_distance_map_heavy_atom(_first_chain, _second_chain):

    # temp_chain_a = fix_res_num_atom(copy.deepcopy(_first_chain))
    # temp_chain_b = fix_res_num_atom(copy.deepcopy(_second_chain))
    temp_chain_a =copy.deepcopy(_first_chain)
    temp_chain_b = copy.deepcopy(_second_chain)
    _first_chain_CA = list(filter(lambda x: (x.atom_name == "CA"), temp_chain_a))
    _second_chain_CA = list(filter(lambda x: (x.atom_name == "CA"), temp_chain_b))

    dist_arry = np.zeros((len(_first_chain_CA),len(_second_chain_CA)))
    contact_arry = np.zeros((len(_first_chain_CA),len(_second_chain_CA)))

    for a_cord in _first_chain_CA:
        _temp_first_chain_data = list(filter(lambda x: (x.res_num == a_cord.res_num), temp_chain_a))

        for b_cord in _second_chain_CA:
            # print(a_cord.res_num,b_cord.res_num)
            _temp_second_chain_data = list(filter(lambda x: (x.res_num == b_cord.res_num), temp_chain_b))
            dist_res = get_distance_resdiues(_temp_first_chain_data,_temp_second_chain_data)
            indx_a=int(a_cord.res_num)-1
            indx_b=int(b_cord.res_num)-1

            dist_arry[indx_a][indx_b] = dist_res
            # if dist_res < INTRA_CHAIN_CONTACT_TRHESHOLD:
            #     contact_arry[indx_a][indx_b] =1

    return dist_arry
class pdb_lines:
    atom = ''
    serial = ''
    atom_name = ''
    alt_loc = ''
    res_name = ''
    chain = ''
    res_num = ''
    icode = ''
    x = ''
    y = ''
    z = ''
    occupancy = ''
    temp_fact = ''
    element = ''
    charge = ''
    visited= 0
    orig_res_num=''
    pass


def read_pdb(pdb):
    contents = []
    with open(pdb, "r") as f:
        for line in f:
            if (line.startswith("ATOM")):
                # pass
                contents.append(line)
    return contents


def remove_residues_without_CA(atom_list):
    # Step 1: find all residues that contain CA
    residues_with_CA = set()
    residues_all = set()

    for atom in atom_list:
        rid = (atom.chain, atom.res_num, atom.res_name)
        residues_all.add(rid)
        if atom.atom_name == "CA":
            residues_with_CA.add(rid)
    residues_without_CA = residues_all - residues_with_CA
    # Step 2: keep only atoms belonging to these residues
    # Print what you filtered out
    if residues_without_CA:
        print("Filtered out residues (no CA atom):")
        for chain, resnum, resname in sorted(residues_without_CA):
            print(f"  {resname} {resnum} (chain {chain})")
    else:
        print("No residues were filtered out. All contained CA.")

    # Now actually filter atoms
    filtered_atoms = [
        atom for atom in atom_list
        if (atom.chain, atom.res_num, atom.res_name) in residues_with_CA
    ]

    return filtered_atoms
# convert_raw_bepi_output_files()


def seq_to_hydrophobicity(seq):
    seq = seq.upper()
    values = [hydrophobicity.get(aa, 0.0) for aa in seq]
    tensor_array= torch.from_numpy(np.array(values, dtype=np.float32)).unsqueeze(dim=1)
    final_hydro_tensor = pad_to_appropitate_L(tensor_array)
    return final_hydro_tensor

def seq_to_charge(seq):
    seq = seq.upper()
    values = [charge.get(aa, 0.0) for aa in seq]
    tensor_array= torch.from_numpy(np.array(values, dtype=np.float32)).unsqueeze(dim=1)
    final_charge_tensor = pad_to_appropitate_L(tensor_array)
    return final_charge_tensor


def get_intra_emdb(_intra_file):
    intra_LxL = load_embedding(_intra_file)  # 1152
    intra_LxL_0 = np.where(intra_LxL<=8,1,0)
    _intra_LxL_pad = pad_to_square(torch.from_numpy(intra_LxL_0))

    return _intra_LxL_pad


def get_intra_distnace(_intra_file):
    intra_LxL = load_embedding(_intra_file)  # 1152

    _intra_LxL_pad = pad_to_square(torch.from_numpy(intra_LxL))

    return _intra_LxL_pad


def contact_density(contact_map):
    """
    contact_map: torch.Tensor [L, L] (0/1)

    Returns:
        relative_packing: [L]
    """
    # Remove self contacts
    contact_map = contact_map.clone()
    contact_map.fill_diagonal_(0)
    # Contacts per residue
    packing = contact_map.sum(dim=1).float()
    # Normalize
    relative_packing = packing / packing.max()
    return relative_packing

def make_template_feature(template, intra, rsa):
    """
    template: [L]
    intra:    [L, L]
    rsa:      [L]

    returns:
        output: [L]
    """

    # 1. Template residues
    template_mask = (template > 0).float() # [L]
    rsa_filtered = rsa.clone()
    rsa_filtered[rsa_filtered < 0.05] = 0
    # 2. Binary contact map
    contact_mask = (intra > 0).float()         # [L, L]

    # 3. Direct neighbors of template residues
    neighbors = (
        contact_mask @ template_mask > 0
    ).float()                                    # [L]

    # 4. Template + its 1-degree neighbors
    region = torch.maximum(
        template_mask,
        neighbors
    )                                            # [L]

    # 5. Contact density
    contact_count = contact_mask.sum(dim=-1)     # [L]

    contact_density = contact_count / (
        contact_count.max() + 1e-8
    )                                            # [L], approximately 0-1

    # 6. Final feature
    output = region.flatten()  * rsa_filtered.flatten()  * (1.0 - contact_density.flatten() )

    return output

def make_template_feature_simple(template, intra):
    """
    template: [L]
    intra:    [L, L]
    rsa:      [L]

    returns:
        output: [L]
    """

    # 1. Template residues
    template_mask = (template > 0).float() # [L]

    contact_mask = (intra > 0).float()         # [L, L]

    # 3. Direct neighbors of template residues
    neighbors = (
        contact_mask @ template_mask > 0
    ).float()                                    # [L]

    # 4. Template + its 1-degree neighbors
    region = torch.maximum(
        template_mask,
        neighbors
    )                                            # [L]

    # 6. Final feature
    output = region.flatten()

    return output

def make_template_loss_mask_no_rsa_cn(template, intra):
    # 1. Template residues
    template_mask = (template > 0).float() # [L]
    # 2. Binary contact map
    contact_mask = (intra > 0).float()         # [L, L]

    # 3. Direct neighbors of template residues
    neighbors = (
        contact_mask @ template_mask > 0
    ).float()                                    # [L]

    region = torch.maximum(
        template_mask,
        neighbors
    )                                            # [L]

    output = template.clone()

    # output[1:] = torch.maximum(output[1:], template[:-1])
    # output[:-1] = torch.maximum(output[:-1], template[1:])
    # output[2:] = torch.maximum(output[2:], template[:-2])
    # output[:-2] = torch.maximum(output[:-2], template[2:])
    # region * (1 - template)
    return   region * (1 - output)




def e_descriptor_from_sequence(seq):
    e = {
        'A': [0.008, 0.134, -0.475, -0.039, 0.181],
        'R': [0.171, -0.361, 0.107, -0.258, -0.364],
        'N': [0.255, 0.038, 0.117, 0.118, -0.055],
        'D': [0.303, -0.057, -0.014, 0.225, 0.156],
        'C': [-0.132, 0.174, 0.07, 0.565, -0.374],
        'Q': [0.149, -0.184, -0.03, 0.035, -0.112],
        'E': [0.221, -0.28, -0.315, 0.157, 0.303],
        'G': [0.218, 0.562, -0.024, 0.018, 0.106],
        'H': [0.023, -0.177, 0.041, 0.28, -0.021],
        'I': [-0.353, 0.071, -0.088, -0.195, -0.107],
        'L': [-0.267, 0.018, -0.265, -0.274, 0.206],
        'K': [0.243, -0.339, -0.044, -0.325, -0.027],
        'M': [-0.239, -0.141, -0.155, 0.321, 0.077],
        'F': [-0.329, -0.023, 0.072, -0.002, 0.208],
        'P': [0.173, 0.286, 0.407, -0.215, 0.384],
        'S': [0.199, 0.238, -0.015, -0.068, -0.196],
        'T': [0.068, 0.147, -0.015, -0.132, -0.274],
        'W': [-0.296, -0.186, 0.389, 0.083, 0.297],
        'Y': [-0.141, -0.057, 0.425, -0.096, -0.091],
        'V': [-0.274, 0.136, -0.187, -0.196, -0.299],
    }

    seq = seq.upper()
    emb = [e.get(aa, [0.0]*5) for aa in seq]
    tensor_array = torch.from_numpy(np.array(emb, dtype=np.float32))
    final_e_tensor = pad_to_appropitate_L(tensor_array)
    return final_e_tensor
def z_descriptor_from_sequence(seq):
    z = {
        'A': [0.07, -1.73, 0.09],
        'R': [2.88, 2.52, -3.44],
        'N': [3.22, 1.45, 0.84],
        'D': [3.64, 1.13, 2.36],
        'C': [0.71, -0.97, 4.13],
        'Q': [2.18, 0.53, -1.14],
        'E': [3.08, 0.39, -0.07],
        'G': [2.23, -5.36, 0.30],
        'H': [2.41, 1.74, 1.11],
        'I': [-4.44, -1.68, -1.03],
        'L': [-4.19, -1.03, -0.98],
        'K': [2.84, 1.41, -3.14],
        'M': [-2.49, -0.27, -0.41],
        'F': [-4.92, 1.30, 0.45],
        'P': [-1.22, 0.88, 2.23],
        'S': [1.96, -1.63, 0.57],
        'T': [0.92, -2.09, -1.40],
        'W': [-4.75, 3.65, 0.85],
        'Y': [-1.39, 2.32, 0.01],
        'V': [-2.69, -2.53, -1.29],
    }

    seq = seq.upper()
    z_embeds = [z.get(aa, [0.0, 0.0, 0.0]) for aa in seq]

    tensor_array = torch.from_numpy(np.array(z_embeds, dtype=np.float32))
    final_z_tensor = pad_to_appropitate_L(tensor_array)
    return final_z_tensor

def weighted_pick(numbers, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()

    return np.random.choice(numbers, p=probabilities)

def random_keep_true_labels_2(label_array, max_keep=4):

    labels = np.array(label_array).copy()
    positive_idx = np.where(labels == 1)[0]
    # Randomly choose how many positives to keep
    # n_keep =  np.random.randint(0, min(max_keep, len(positive_idx)) + 1)

    n_keep_2 = weighted_pick(
        numbers=[0, 1, 2, 3,4,np.floor(len(positive_idx)/10 ) , np.floor(len(positive_idx)/5 )  ],
        probabilities=[0.1, 0.1, 0.1,  0.05, 0.05, 0.1, 0.1]
    )
    if max_keep == 0 :
        n_keep_2 = 0
    else:
        try:
            n_keep_2 = min(int(n_keep_2),len(positive_idx) )
        except:
            n_keep_2 = 0


    # Randomly select positives
    keep_idx = np.random.choice(
        positive_idx,
        size=n_keep_2,
        replace=False
    ) if n_keep_2 > 0 else []

    # Create new masked array
    masked = np.zeros_like(labels)

    masked[keep_idx] = 1

    return masked


def random_keep_true_labels(label_array, max_keep=4):

    labels = np.array(label_array).copy()
    positive_idx = np.where(labels == 1)[0]
    # Randomly choose how many positives to keep
    n_keep = np.random.randint(0, min(max_keep, len(positive_idx)) + 1)

    # Randomly select positives
    keep_idx = np.random.choice(
        positive_idx,
        size=n_keep,
        replace=False
    ) if n_keep > 0 else []

    # Create new masked array
    masked = np.zeros_like(labels)

    masked[keep_idx] = 1

    return masked

MAX_ASA = {
    'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
    'Q': 225, 'E': 223, 'G': 104, 'H': 224, 'I': 197,
    'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
    'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174
}


def sasa_to_rsa(sequence, sasa_values, clip=True):
    sequence = sequence.upper()
    if len(sequence) != len(sasa_values):
        raise ValueError(
            f"Sequence length ({len(sequence)}) and SASA length "
            f"({len(sasa_values)}) do not match."
        )
    rsa_values = []
    for residue, sasa in zip(sequence, sasa_values):
        if residue not in MAX_ASA:
            raise ValueError(f"Unknown amino acid: {residue}")
        rsa = float(sasa) / MAX_ASA[residue]
        if clip:
            rsa = min(rsa, 1.0)
        rsa_values.append(rsa)
    rsa_values_final = pad_to_appropitate_L(torch.from_numpy(np.asarray(rsa_values)).unsqueeze(-1))
    return rsa_values_final


import torch
import torch.nn.functional as F

def bin_distance(distance_matrix, seq_len):

    bins = torch.tensor(
        [2.,4.,6.,8.,10.,12.,14.,16.,18.,20.],
        device=distance_matrix.device
    )

    bin_index = torch.bucketize(distance_matrix, bins)

    one_hot = F.one_hot(bin_index, num_classes=11).float()

    # Zero padded area
    one_hot[seq_len:, :, :] = 0
    one_hot[:, seq_len:, :] = 0

    return one_hot