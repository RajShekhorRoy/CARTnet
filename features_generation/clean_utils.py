import numpy as np
import copy
import os
import subprocess
from Bio import pairwise2

CONTACT_THRESHOLD = 6
def remove_duplicate_ca(atoms):
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
    
def separate_by_chain(_pdb, _name):
    # print(_pdb)
    result = list(filter(lambda x: (x.chain == _name), _pdb))
    return result


def chain_mapper(_a_chains, _b_chains):
    length_chain_a = len(_a_chains)
    __a_chains = _a_chains.replace(":", " ")
    __b_chains = _b_chains.replace(":", " ")
    ovr_chn_a = []
    ovr_chn_b = []
    ovr_comb_chaim = []
    chain_map = {}
    for i in range(length_chain_a):
        a_chain = __a_chains[i]
        b_chain = __b_chains[i]
        if a_chain != " " and b_chain != " ":
            ovr_chn_a.append(str(a_chain))
            ovr_chn_b.append(str(b_chain))
            # ovr_comb_chaim.append(str(a_chain) + "_" + str(b_chain))
            chain_map[a_chain] = b_chain
    return chain_map, ovr_chn_a, ovr_chn_b


def get_all_interactions(_chains, _profile):
    chains_1 = _chains
    iq_profile_obj = _profile
    interactions = {}
    #### Find all the interfaces in the original
    for i in range(len(chains_1)):
        temp_array = []
        for j in range(len(chains_1)):
            if i != j + 1 and len(chains_1) > j + 1 > i:
                if if_contact(iq_profile_obj.original_ref_pdb[chains_1[i]],
                              iq_profile_obj.original_ref_pdb[chains_1[j + 1]]):
                    temp_array.append(chains_1[j + 1])
                    # print(chains_1[j + 1])
                    # print(chains_1[i], chains_1[j + 1])

        interactions[chains_1[i]] = temp_array

    return interactions


class iq_profile:
    original_ref_fasta = {}
    original_com_fasta = {}

    overlapping_fastas = {}

    original_ref_pdb = {}
    original_com_pdb = {}

    original_ref_pdb_ca = {}
    original_com_pdb_ca = {}

    aligned_ref_fasta = {}
    aligned_com_fasta = {}

    aligned_pdb_ref_ca = {}
    aligned_pdb_com_ca = {}

    ref_interactions = {}
    com_interactions = {}

    distance_maps_ref = {}
    distance_maps_com = {}

    contact_maps_ref = {}
    contact_maps_com = {}
    pass


# from PIL import Image as im

def get_MM_score(_arr):
    MM_ALIGN_PATH = _arr[2]
    # print(MM_ALIGN_PATH, _arr[0], _arr[1])
    contents = subprocess.check_output([MM_ALIGN_PATH, _arr[0], _arr[1]])
    try:
        tm_list = []

        for item in contents.decode("utf-8").split("\n"):

            if "TM-score=" in item:
                tm_list.append(float(item.strip().split(" ")[1].strip()))

        return np.min(tm_list)
    except:
        return 0.0


def read_pdb(pdb):
    contents = []
    with open(pdb, "r") as f:
        for line in f:
            if (line.startswith("ATOM")):
                # pass
                contents.append(line)
    return contents


def space_returner(_input):
    i = 0
    space = ""
    while i < _input:
        space = space + " "
        i = i + 1
    return space


fasta_3_to_1_code = {'ala': 'A', 'arg': 'R', 'asn': 'N', 'asp': 'D', 'asx': 'B', 'cys': 'C', 'glu': 'E', 'gln': 'Q',
                     'glx': 'Z', 'gly': 'G', 'his': 'H', 'ile': 'I', 'leu': 'L', 'lys': 'K', 'met': 'M', 'phe': 'F',
                     'pro': 'P', 'ser': 'S', 'thr': 'T', 'trp': 'W', 'tyr': 'Y', 'val': 'V'}


def find_lowest_gap(_target, _hit):
    aln_val = pairwise2.align.globalms(_target, _hit, 5, -4, -1, -0.1)
    chain_target = list(aln_val[0][0])
    chain_hit = list(aln_val[0][1])

    return chain_hit.count('-') / len(chain_hit)


def added_warning_logs(_file, _msg):
    # Open a file with access mode 'a'
    file_object = open(_file, 'a')
    # Append 'hello' at the end of file
    file_object.write(_msg)
    # Close the file
    file_object.close()
    return


def specific_filename_reader(_input_dir, _extension):
    file_names = []
    for root, directories, files in os.walk(_input_dir):
        for file in files:
            if _extension in file:
                file = file.split(".")[0]
                if not file in file_names:
                    file_names.append(file.split(".")[0])
    return file_names


def dir_maker(_dir_name):
    if not os.path.exists(_dir_name):
        os.system("mkdir -p " + _dir_name)
        return _dir_name
    else:
        print("Already exists ")
        return _dir_name

def txt_appender(_filename,_text):
    with open(_filename, "a") as f:
        f.writelines(_text)
        
    print(_filename)
    return

def convert_to_pdb(_pdb, _name):
    content = ''
    for x in _pdb:
        content += correct_format(x) + '\n'
    f = open(_name, "w")
    f.write(content)
    f.close()
    return _pdb


def find_common_fasta(_target, _hit):
    aln_val = pairwise2.align.globalms(_target, _hit, 5, -4, -1, -0.1)
    chain_target = list(aln_val[0][0])
    chain_hit = list(aln_val[0][1])
    # print(chain_target)
    # print(chain_hit)
    common_fasta = ""
    for counter in range(0, len(chain_hit)):
        if chain_target[counter] == chain_hit[counter]:
            common_fasta += chain_target[counter]
    return aln_val[0].seqA, aln_val[0].seqB, common_fasta


def closest_key(_seq_fasta_dict, _fasta_string):
    val = []
    for key in _seq_fasta_dict:
        val.append(find_lowest_gap(_seq_fasta_dict.get(key), _fasta_string))
    seq = min(val)
    index_closest = val.index(seq)
    return index_closest


def sequence_finder(_seq_fasta_dict, _fasta_string):
    for key in _seq_fasta_dict:
        temp_fasta = _seq_fasta_dict.get(key)
        if temp_fasta == _fasta_string:
            return key
    seq_ = closest_key(_seq_fasta_dict, _fasta_string)
    # print(" closest_key ")
    return seq_


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


def string_array_from_pdb_array(_pdb_row):
    _pdb_copy = copy.deepcopy(_pdb_row)
    # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    _pdb_copy.atom = _pdb_copy.atom  # 1-4
    _pdb_copy.serial = space_returner(4 - len(str(_pdb_copy.serial))) + str(_pdb_copy.serial)  # 7-11
    _pdb_copy.atom_name = _pdb_copy.atom_name + space_returner(3 - len(_pdb_copy.atom_name))  # 13-16
    _pdb_copy.alt_loc = space_returner(1 - len(_pdb_copy.alt_loc)) + _pdb_copy.alt_loc  # 17
    _pdb_copy.res_name = space_returner(3 - len(_pdb_copy.res_name)) + _pdb_copy.res_name  # 18-20
    _pdb_copy.chain = space_returner(1 - len(_pdb_copy.chain)) + _pdb_copy.chain  # 22
    _pdb_copy.res_num = space_returner(4 - len(_pdb_copy.res_num)) + _pdb_copy.res_num  # 23-26
    # _pdb_copy.icode = space_returner(2 - len(_pdb_copy.chain)) + _pdb_copy.icode  # 27
    _pdb_copy.icode = space_returner(1 - len(_pdb_copy.icode)) + _pdb_copy.icode  # 27
    _pdb_copy.x = space_returner(8 - len(_pdb_copy.x)) + _pdb_copy.x  # 31-38
    _pdb_copy.y = space_returner(8 - len(_pdb_copy.y)) + _pdb_copy.y  # 39-46
    _pdb_copy.z = space_returner(8 - len(_pdb_copy.z)) + _pdb_copy.z  # 47-54
    _pdb_copy.occupancy = space_returner(6 - len(_pdb_copy.occupancy)) + _pdb_copy.occupancy  # 55-60
    _pdb_copy.temp_fact = space_returner(6 - len(_pdb_copy.temp_fact)) + _pdb_copy.temp_fact  # 61-66
    _pdb_copy.element = space_returner(4 - len(_pdb_copy.element)) + _pdb_copy.element  # 73-76
    _pdb_copy.charge = space_returner(2 - len(_pdb_copy.charge)) + _pdb_copy.charge  # 77-78

    content = _pdb_copy.atom + space_returner(7 - len(_pdb_copy.serial)) + _pdb_copy.serial
    if len(_pdb_copy.atom_name) < 4:
        content = content + space_returner(2) + _pdb_copy.atom_name
    elif len(_pdb_copy.atom_name) == 4:
        content = content + " " + _pdb_copy.atom_name

    content = content + _pdb_copy.alt_loc + _pdb_copy.res_name + space_returner(
        1) + _pdb_copy.chain + _pdb_copy.res_num + _pdb_copy.icode + space_returner(
        3) + _pdb_copy.x + _pdb_copy.y + _pdb_copy.z + _pdb_copy.occupancy + _pdb_copy.temp_fact + space_returner(
        8) + _pdb_copy.element + _pdb_copy.charge
    return content


def pdb_from_array(_pdb, _filename):
    array = []
    content = ''
    number = 1
    for x in _pdb:
        x.serial = number
        val = string_array_from_pdb_array(x)
        array.append(val)
        content = content + val + '\n'
        number = number + 1
    f = open(_filename, "w")
    f.write(content + 'END')
    f.close()
    return array
def pdb_from_array_multi_chain(_pdb, _filename):
    array = []
    content = ''
    number = 1
    add_ter=False
    prev_chain=_pdb[0].chain
    for x in _pdb:
        if prev_chain != x.chain :
            content = content + "TER" + '\n'
            prev_chain = x.chain
        x.serial = number
        val = string_array_from_pdb_array(x)
        array.append(val)
        content = content + val + '\n'
        number=number+1
        # if add_ter ==True:
        #     content=content+"TER"+ '\n'
        #     add_ter=False
    f = open(_filename, "w")
    f.write(content + 'TER\nEND')
    f.close()
    return array

def contents_to_info(contents):  # reads the ATOM line. Then splits the info into respective frames and returns the data
    split_contents = []
    for lines in contents:
        if lines.startswith("ATOM"):
            pdb_line = split_line_to_tuple(lines.strip())
            split_contents.append(pdb_line)
    return split_contents

def fix_serial(_array, _no=1):
    number = _no
    for x in _array:
        x.serial = number
        number = number + 1
    return _array


def get_CA_cmaps(_first_chain, _second_chain):
    CA_first_chain = list(filter(lambda x: (x.atom_name == "CA"), copy.deepcopy(_first_chain)))
    CA_second_chain = list(filter(lambda x: (x.atom_name == "CA"), copy.deepcopy(copy.deepcopy(_second_chain))))
    chain_len_a = len(CA_first_chain)
    chain_len_b = len(CA_second_chain)
    cmap_array = np.zeros((chain_len_a, chain_len_b))
    for a_cord in range(chain_len_a):
        for b_cord in range(chain_len_b):
            # can make it a little more optimized
            dist = np.sqrt((float(CA_first_chain[a_cord].x) - float(CA_second_chain[b_cord].x)) ** 2 + (
                    float(CA_first_chain[a_cord].y) - float(CA_second_chain[b_cord].y)) ** 2 + (
                                   float(CA_first_chain[a_cord].z) - float(CA_second_chain[b_cord].z)) ** 2)
            if dist <= 8:
                cmap_array[a_cord][b_cord] = 1

    return cmap_array


def if_contact(_first_chain_path, _second_chain_path):
    first_chain_CA = list(filter(lambda x: (x.atom_name == "CA"), _first_chain_path))
    second_chain_CA = list(filter(lambda x: (x.atom_name == "CA"), _second_chain_path))
    for a_cord in first_chain_CA:
        for b_cord in second_chain_CA:
            # can make it a little more optimized
            if np.sqrt((float(a_cord.x) - float(b_cord.x)) ** 2 + (float(a_cord.y) - float(b_cord.y)) ** 2 + (
                    float(a_cord.z) - float(b_cord.z)) ** 2) <= CONTACT_THRESHOLD:
                return True
    return False


def distance(coord1, coord2):
    x1 = float(coord1["x"])
    y1 = float(coord1["y"])
    z1 = float(coord1["z"])
    x2 = float(coord2["x"])
    y2 = float(coord2["y"])
    z2 = float(coord2["z"])
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    return d


def correct_format(_pdb_row):
    _pdb_copy = copy.deepcopy(_pdb_row)
    # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    _pdb_copy.atom = _pdb_copy.atom  # 1-4
    _pdb_copy.serial = space_returner(5 - len(str(_pdb_copy.serial))) + str(_pdb_copy.serial)  # 7-11
    _pdb_copy.atom_name = _pdb_copy.atom_name + space_returner(3 - len(_pdb_copy.atom_name))  # 13-16
    _pdb_copy.alt_loc = space_returner(1 - len(_pdb_copy.alt_loc)) + _pdb_copy.alt_loc  # 17
    _pdb_copy.res_name = space_returner(3 - len(_pdb_copy.res_name)) + _pdb_copy.res_name  # 18-20
    _pdb_copy.chain = space_returner(1 - len(_pdb_copy.chain)) + _pdb_copy.chain  # 22
    _pdb_copy.res_num = space_returner(4 - len(_pdb_copy.res_num)) + _pdb_copy.res_num  # 23-26
    # _pdb_copy.icode = space_returner(2 - len(_pdb_copy.chain)) + _pdb_copy.icode  # 27
    _pdb_copy.icode = space_returner(1 - len(_pdb_copy.icode)) + _pdb_copy.icode  # 27
    _pdb_copy.x = space_returner(8 - len(_pdb_copy.x)) + _pdb_copy.x  # 31-38
    _pdb_copy.y = space_returner(8 - len(_pdb_copy.y)) + _pdb_copy.y  # 39-46
    _pdb_copy.z = space_returner(8 - len(_pdb_copy.z)) + _pdb_copy.z  # 47-54
    _pdb_copy.occupancy = space_returner(6 - len(_pdb_copy.occupancy)) + _pdb_copy.occupancy  # 55-60
    _pdb_copy.temp_fact = space_returner(6 - len(_pdb_copy.temp_fact)) + _pdb_copy.temp_fact  # 61-66
    _pdb_copy.element = space_returner(4 - len(_pdb_copy.element)) + _pdb_copy.element  # 73-76
    _pdb_copy.charge = space_returner(2 - len(_pdb_copy.charge)) + _pdb_copy.charge  # 77-78
    content = _pdb_copy.atom + space_returner(2) + _pdb_copy.serial

    if len(_pdb_copy.atom_name) < 4:
        content = content + space_returner(2) + _pdb_copy.atom_name
    elif len(_pdb_copy.atom_name) == 4:
        content = content + " " + _pdb_copy.atom_name

    content = content + _pdb_copy.alt_loc + _pdb_copy.res_name + space_returner(
        1) + _pdb_copy.chain + _pdb_copy.res_num + _pdb_copy.icode + space_returner(
        3) + _pdb_copy.x + _pdb_copy.y + _pdb_copy.z + _pdb_copy.occupancy + _pdb_copy.temp_fact + space_returner(
        8) + _pdb_copy.element + _pdb_copy.charge

    return content


def write2File(_filename, _cont):
    with open(_filename, "w") as f:
        f.writelines(_cont)
        f.close()


def get_skeleton(_pdb):
    return list(filter(lambda x: (x.atom_name == "CA"), _pdb))


def get_unique_chains(_inp_details):
    chain_array = []
    for val in _inp_details:
        chain_array.append(val.chain)
    return list(dict.fromkeys(chain_array))


def get_fasta_from_pdb_array(_pdb, _chain):
    pdb_a = separate_by_chain(copy.deepcopy(_pdb), _chain)
    index_tracker_a = []
    for val in pdb_a:
        index_tracker_a.append(str(val.res_num) + "_" + str(val.res_name))
    index_tracker_a = list(dict.fromkeys(index_tracker_a))
    # print(index_tracker_a)
    fasta_string = ''
    for values in index_tracker_a:
        three_code = values.split('_')[1].lower()
        fasta_string = fasta_string + str(fasta_3_to_1_code.get(three_code))
    # print(len(fasta_string))
    return fasta_string


def monomer_pdb_filtering(_pdb, _dir):
    tar_name = os.path.basename(_pdb)
    tar_dir = _dir + "/" + tar_name
    os.system("mkdir -p " + tar_dir)
    full_pdb = contents_to_info(read_pdb(_pdb))
    chain_finder = get_unique_chains(full_pdb)
    # print(chain_finder)
    for chain in chain_finder:
        temp_monomer_pdb = separate_by_chain(full_pdb, chain)
        tar_monomer_file = tar_dir + "/" + tar_name + "_chain_" + str(chain) + ".pdb"
        monomer_string = pdb_from_array(_pdb=temp_monomer_pdb, _filename=tar_monomer_file)

        fasta_name = get_fasta_from_pdb_array(temp_monomer_pdb)
        fasta_value = ">sequence_" + chain + "\n" + fasta_name
        fasta_file_name = tar_dir + "/" + tar_name + "_chain_" + str(chain) + ".fasta"
        write2File(_filename=fasta_file_name, _cont=fasta_value)

    return chain_finder

def fix_res_num_atom(_pdb):
    ca_index = 0
    prev_tag = ""
    # _pdb_ = increase_residue_n(copy.deepcopy(copy.deepcopy(_pdb)))
    for values in _pdb:
    
        current_tag = values.res_name + '_' + str(values.orig_res_num)+"_"+values.icode
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

            prev_tag = values.res_name + '_' + str(values.orig_res_num)+"_"+values.icode
    _pdb = fix_serial(_pdb)
    return _pdb


def get_distance_map(first_chain_CA, second_chain_CA):
    dist_arry = np.zeros((len(first_chain_CA), len(second_chain_CA)))

    contact_arry = np.zeros((len(first_chain_CA), len(second_chain_CA)))

    for a_cord in fix_res_num_atom(first_chain_CA):
        for b_cord in fix_res_num_atom(second_chain_CA):
            value = np.sqrt((float(a_cord.x) - float(b_cord.x)) ** 2 + (float(a_cord.y) - float(b_cord.y)) ** 2 + (
                    float(a_cord.z) - float(b_cord.z)) ** 2)
            dist_arry[int(a_cord.res_num) - 1][int(b_cord.res_num) - 1] = value
            if value < CONTACT_THRESHOLD:
                contact_arry[int(a_cord.res_num) - 1][int(b_cord.res_num) - 1] = 1

    return dist_arry, contact_arry

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

def get_distance_map_heavy_atom(_first_chain, _second_chain):

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
            if dist_res < CONTACT_THRESHOLD:
                contact_arry[indx_a][indx_b] =1

    return dist_arry,contact_arry

def mark_matrix(seq_a, seq_b, matrix):
    matrix = np.array(matrix)  # Convert to string for marking

    # Remove '-' from seq_a and seq_b to get the original dimensions
    clean_seq_a = [char for char in seq_a if char != '-']
    clean_seq_b = [char for char in seq_b if char != '-']

    # Create an empty adjusted matrix of correct dimensions
    adjusted_matrix = np.array(matrix[:len(clean_seq_a), :len(clean_seq_b)])

    # Reintroduce '-' in seq_a by mapping original indices
    row_indices = [i for i, char in enumerate(seq_a) if char == '-']

    # Reintroduce '-' in seq_b by mapping original indices
    col_indices = [j for j, char in enumerate(seq_b) if char == '-']

    # Mark the matrix
    for i in row_indices:
        adjusted_matrix = np.insert(adjusted_matrix, i, 9999, axis=0)
    for j in col_indices:
        adjusted_matrix = np.insert(adjusted_matrix, j, 9999, axis=1)

    return adjusted_matrix


def calculate_rmsd_ignore_mask(array1, array2, mask_value=9999):
    """Compute the RMSD between two 2D NumPy arrays, ignoring cells with the mask_value."""
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape to compute RMSD.")

    # Create a mask for valid values (i.e., not equal to mask_value in either array)
    valid_mask = (array1 != mask_value) & (array2 != mask_value)

    # Extract valid values
    valid_array1 = array1[valid_mask]
    valid_array2 = array2[valid_mask]

    # Compute RMSD
    if valid_array1.size == 0:  # If no valid values, return 0
        return 0.0

    return np.sqrt(np.mean((valid_array1 - valid_array2) ** 2))


def get_aligned_distmaps(_dist_ori, _dist_com, _aln_ref_chain_a, _aln_com_chain_a, _aln_ref_chain_b, _aln_com_chain_b):
    new_ori_cmap = np.zeros((len(_aln_ref_chain_a), len(_aln_ref_chain_b)))
    new_com_cmap = np.zeros((len(_aln_com_chain_a), len(_aln_com_chain_b)))

    _new_dist_ori_map = mark_matrix(_aln_ref_chain_a, _aln_ref_chain_b, _dist_ori)
    _new_dist_com_map = mark_matrix(_aln_com_chain_a, _aln_com_chain_b, _dist_com)
    print(calculate_rmsd_ignore_mask(_new_dist_ori_map,_new_dist_com_map))
    return new_ori_cmap, new_com_cmap

def get_fasta_format(_name,_fasta):
    return ">{0}\n{1}".format(_name,_fasta)

def get_str_from_array(_arr):
    out_str = ""
    for values in _arr:
        out_str+=str(values)+","
    return out_str

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

def multi_fasta_reader(_seq_file):
    file = open(_seq_file, "r")
    stoi_fasta_dict = {}
    output_array = []
    if file.mode == 'r':
        output_array = file.read().strip().splitlines()
        file.close()
    counter = 0
    temp_name = ""
    for values in output_array:
        if counter % 2 == 1:
            stoi_fasta_dict[temp_name] = values
        else:
            temp_name = str(values)
        counter = counter + 1

    return stoi_fasta_dict

def fasta_reader(_seq_file):
    file = open(_seq_file, "r")
    output_array = []
    if file.mode == 'r':
        output_array = file.read().splitlines()
        file.close()
    return output_array[1]

def text_file_reader(_file):
    file = open(_file, "r")
    output_array = []
    if file.mode == 'r':
        output_array = file.read().splitlines()
    file.close()

    return output_array

