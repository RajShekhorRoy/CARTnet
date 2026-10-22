import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import libs
import os

import torch
from pytorch_lightning import   seed_everything
from libs import text_file_reader,  save_esm_embedding
from utils import  get_sequence_identity_features, get_esm_embedding, get_labels_BP3, \
    first_x_ones,   seq_to_hydrophobicity, \
    seq_to_charge, get_intra_emdb,  random_keep_true_labels

from model import CARTNET


MAX_LENGTH = 1400
seed_everything(42, workers=True)


BATCH_SIZE =1
nThreads = 0
TEMPLATE_COUNT = 0
ID= "TESTSETSET"
# checkpoint_file =   '/DATA/EPITOPE/BENCHMARK/my_networks/attentiong_net/net5M_01102026/checkpoints/epochepoch={0}.ckpt'.format(ID)
checkpoint_file = "/home/rajroy/PycharmProjects/CARTnet/model/model_weights/RES_NO_Temp/RM_NOTemp_FINAL_swiglu082726_lr_EXT20_082626epoch=026.ckpt"
model = CARTNET.CARTnet.load_from_checkpoint(checkpoint_file)
model.eval()
output_dir="/home/rajroy/result_dump/test/8UDG/".format(ID)
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__target ="8UDG_A"
esm_file = "/home/rajroy/result_dump/test/8UDG_A/8UDG_A_struture_structure_embed.npy"
fasta_file = "/home/rajroy/result_dump/test/8UDG_A/8UDG_A_struture.fasta"
ss8_file = "/home/rajroy/result_dump/test/8UDG_A/8UDG_A_struture_dssp.npy"
intra_file = "/home/rajroy/result_dump/test/8UDG_A/8UDG_A_struture_dist.npz.npy"

try:

    sequence_fasta = text_file_reader(fasta_file)[1]
    sequence_length = len(sequence_fasta)
    #work here
    template_fasta = sequence_fasta
    seq_identity_feature_pad = get_sequence_identity_features(fasta_file)

    esm_feature_pad = get_esm_embedding(esm_file) # 1152
    feat = torch.tensor(first_x_ones(sequence_length, MAX_LENGTH)).unsqueeze(dim=1)
    esm_feature_pad = torch.cat([esm_feature_pad, feat], dim=1)

    sasa = get_esm_embedding(ss8_file)[:, 1:9]
    charge_t = seq_to_charge(sequence_fasta.upper())
    hydro_t = seq_to_hydrophobicity(sequence_fasta.upper())

    labels_pad = get_labels_BP3(template_fasta)

    intra = get_intra_emdb(intra_file)
    x4=intra = torch.tensor(intra, dtype=torch.float32).cuda()
    function_feat = torch.tensor(torch.cat([seq_identity_feature_pad, sasa, charge_t, hydro_t, feat], axis=1),
                                 dtype=torch.float32)
    x3=function_feat = function_feat.permute(0, 1).unsqueeze(0).cuda()
    x2=esm_feature_pad = esm_feature_pad.permute(1, 0).unsqueeze(0).cuda()

    name =  __target
#changes based on tempalte
    x5=template_pad = torch.tensor(random_keep_true_labels(label_array=labels_pad, max_keep=TEMPLATE_COUNT)).unsqueeze(dim=0).cuda()
    labels_pad = labels_pad.unsqueeze(0)
    logits = model(x2.unsqueeze(dim=0),x3.unsqueeze(dim=0),x4.unsqueeze(dim=0),x5.unsqueeze(dim=0))
    content =logits.cpu().detach().numpy()
    # write2File(output_dir+str(__target)+"predicted_labels.txt",content)
    save_esm_embedding(logits,output_dir+str(__target)+"predicted_labels.npy")
    print(logits)
except :
    print(__target)