import os

from libs import  write2File
from libs import text_file_reader,  fasta_file_reader
from utils import caps_at_indexes
counter_list  = []
total_list = []

target_id = "/home/rajroy/result_dump/test/temp_test/6VYV_PLH.pdb"
output_target_dir = "/home/rajroy/result_dump/test/temp_test/"

fasta_dir =output_target_dir+"/antigen_fasta/"
os.makedirs(fasta_dir,exist_ok=True)
label_dir =output_target_dir+"/labels/"
os.makedirs(label_dir,exist_ok=True)
output_dir =output_target_dir+"/label_fasta/"
os.makedirs(output_dir,exist_ok=True)

final_str = ""

pdb_id = os.path.basename(target_id).split(".")[0].split("_")[0]
antigen_chain_id = os.path.basename(target_id).split("_")[1].split(".")[0][-1]
antibody_chain_id =os.path.basename(target_id).split("_")[1].split(".")[0][0:-1]

list_labels=["{0}_{1}_{2}_interacting_residues_distance".format(pdb_id,antibody_chain_id,antigen_chain_id)]
list_seqs = ["{0}_{1}".format(pdb_id,antigen_chain_id)]
for values in list_seqs:
    total_list.append(values)
    antigent_id = values.split("_")[0]
    antigent_chain = values.split("_")[1]
    short_list = [x for x in list_labels if antigent_id in x]
    antigen_label_file =  ""
    for id in short_list:
        temp_id = id.split("_")[2]
        if temp_id ==antigent_chain :
            antigen_label_file = label_dir+"{0}.txt".format(id)
            break

    label_index_1 = text_file_reader(antigen_label_file)
    label_index_0 = [int(x)-1 for x in label_index_1[0].split(",") if x.strip()]
    fasta = fasta_file_reader(fasta_dir + str(values) + ".fasta").lower()
    if len(label_index_0) != 0:
        label_string = caps_at_indexes(fasta, label_index_0)
    output_file = output_dir + str(values) + "_label.txt"
    fasta_format =">{0}\n{1}".format(values,label_string)
    write2File(output_file, fasta_format)
    final_str = final_str+"{0} , {1}, {2} \n".format(values, len(label_index_0),len(fasta))
    counter_list.append(values)


