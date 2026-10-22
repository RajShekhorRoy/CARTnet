import numpy as np
propenisity_dict= {"A":1.064,"C":1.412,"D":0.866,"E":0.851,"F":1.091,"G":0.874,"H":1.105,
                   "I":1.152,"K":0.93,"L":1.25,"M":0.826,"N":0.776,"P":1.064,"Q":1.105,
                   "R":0.873,"S":1.012,"T":0.909,"V":1.383,"W":0.893,"Y":1.161}
def seq_to_hydrophobicity(seq, hydrophobicity_dict):
    seq = seq.upper()
    values = [hydrophobicity_dict.get(aa, 0.0) for aa in seq]
    return np.array(values, dtype=np.float32)

def get_seq_propensity(_seq):
    embed=[]
    _seq= _seq.upper()
    for values in _seq:
        embed.append(propenisity_dict[values])

    return embed

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

import numpy as np

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

    return np.array(emb, dtype=np.float32)
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

    return np.array(z_embeds, dtype=np.float32)

def sliding_avg(arr, window=7):
    arr = np.array(arr)
    n = len(arr)
    half = window // 2

    result = np.zeros(n)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)

        result[i] = arr[start:end].mean()

    return result
def uppercase_indexes(s):
    return [i for i, ch in enumerate(s) if ch.isupper()]
def get_labels_BP3 (_seq):
    data_label = uppercase_indexes(_seq)
    L= len(_seq)
    _array = np.zeros(L)
    _array[data_label] = 1
    _labels_pad =    _array
    return _labels_pad

def text_file_reader(_file):
    file = open(_file, "r")
    output_array = []
    if file.mode == 'r':
        output_array = file.read().splitlines()
    file.close()

    return output_array

def load_embedding(load_path ):
    emb = np.load(load_path )
    return emb

def get_20_value(_arr):
    sorted_arr = sorted(_arr)
    cutoff_index = int(0.3 * len(sorted_arr))
    threshold = sorted_arr[cutoff_index]
    return threshold

def top_20_percent_binary(arr):

    threshold = get_20_value(arr)

    return [1 if x >= threshold else 0 for x in arr]

from scipy.stats import pearsonr

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
def correlation_scipy(x, y):
    corr, p_value = pearsonr(x, y)
    return corr, p_value

_fasta_dir = "/DATA/EPITOPE/CLEANED_AACDB_DATA/No_length_limit_AACDB_CLEANED/epitope_collapse/EC_95/"
train_list ,val_list,test_list  = ("/home/rajroy/PycharmProjects/Epitope/dataset/C05/original/my_train_ids_ep95.txt",
                                   "/home/rajroy/PycharmProjects/Epitope/dataset/C05/original/my_val_ids_ep95.txt","/home/rajroy/PycharmProjects/Epitope/dataset/C05/original/my_test_ids_ep95.txt")

sasa_dir = "/DATA/EPITOPE/DATASET_FEATURES/NEW_DATA_AACDB_1400_05/dssp/"
train_file_list = text_file_reader(train_list)
test_file_list = text_file_reader(test_list)
val_file_list = text_file_reader(val_list)
all_file = [*train_file_list, *test_file_list, *val_file_list]
miss_counter= 0
all_corelat = []
# all_file.remove('7UYM_K')
for value in all_file:

    sasa = sasa_dir+"{0}.npy".format(value)
    fasta_file = _fasta_dir + str(value) + "_labels_021626.txt"

    esm_file = "/DATA/EPITOPE/DATASET_FEATURES/NEW_DATA_AACDB_1400_05/structure_embeddings/{0}_structure_embed.npy".format(value)
    esm =load_embedding(esm_file)
    sequence_fasta = text_file_reader(fasta_file)[1]
    sava_val = load_embedding(sasa)[:,0]
    # sava_val=np.asarray(top_20_percent_binary (sava_val))

    contact_pact = "/DATA/EPITOPE/DATASET_FEATURES/NEW_DATA_AACDB_1400_05/intrachain_distance/"+"{0}_dist.npz.npy".format(value)
    contact_pact_val = load_embedding(contact_pact)

    # Step 1 & 2: convert >8 → 0, else → 1
    binary = np.where(contact_pact_val > 8, 0, 1)

    # Step 3 & 4: row-wise average → 1D array
    row_avg =1- ( binary.mean(axis=1)/binary.shape[1])*500


    # sequence_fasta = text_file_reader(fasta_file)[1]

    # try:
    #     # sava_val_seq = sliding_avg(np.asarray(get_seq_propensity(sequence_fasta)))
    #     # sava_val =  seq_to_hydrophobicity(sequence_fasta, charge)
    # except:
    #     print(sequence_fasta)

    labels_pad = get_labels_BP3(sequence_fasta)

    if labels_pad.shape!= sava_val.shape:

        print(value,labels_pad.shape,sava_val.shape[0],esm.shape)

        miss_counter += 1
    else:
        if np.sum(sava_val)==0:
            continue
        sava_val = sava_val * row_avg
        corelat = correlation_scipy(sava_val, labels_pad)
        # print(corelat)

        all_corelat.append(corelat)




print(np.average(np.asarray(all_corelat)))