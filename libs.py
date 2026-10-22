import os
from pathlib import Path
import torch
import numpy as np

def fasta_file_reader(_filename):
    return file_reader(_filename)[1]


def file_reader(_filename):
    file = open(_filename, "r")
    __output_array=[]
    if file.mode == 'r':
        __output_array = file.read().strip().splitlines()
        file.close()
    return __output_array

def specific_filename_reader(_input_dir, _extension):
    file_names = []
    for root, directories, files in os.walk(_input_dir):
        for file in files:
            if _extension in file:
                file = file.split(".")[0]
                if not file in file_names:
                    file_names.append(file.split(".")[0])
    return file_names

def save_esm_embedding(embedding, save_path):

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if type(embedding) is torch.Tensor:
        # Always detach and move to CPU before saving
        embedding = embedding.detach().cpu().numpy()

    np.save(save_path,embedding )

def fasta_reader(_seq_file):
    file = open(_seq_file, "r")
    output_array = []
    if file.mode == 'r':
        output_array = file.read().splitlines()
        file.close()
    return output_array[1]
def multi_fasta_reader(_seq_file):

    stoi_fasta_dict = {}
    output_array = file_reader(_seq_file)
    counter = 0
    temp_name = ""
    for values in output_array:
        if counter % 2 == 1:
            stoi_fasta_dict[temp_name] = values
        else:
            temp_name = str(values)
        counter = counter + 1

    return stoi_fasta_dict


def write2File(_filename, _cont):
    with open(_filename, "w") as f:
        f.writelines(_cont)
        f.close()
    return


def text_file_reader(_file):
    file = open(_file, "r")
    output_array = []
    if file.mode == 'r':
        output_array = file.read().splitlines()
    file.close()

    return output_array


def check_single_exists(_file):
    if os.path.exists(_file):
        return True
    else:
        return False



def write2File_from_array(_filename, _cont):
    # os.makedirs(_filename,exist_ok=True)
    print_string = ""
    for values in _cont:
        print_string += str(values)+"\n"
    with open(_filename, "w") as f:
        f.writelines(print_string)
        f.close()
    return
