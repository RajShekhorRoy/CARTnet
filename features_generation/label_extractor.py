import numpy as np
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT = os.path.dirname(ROOT)
# sys.path.insert(0, ROOT)
import os
import copy
import clean_utils

# _input_struct_pdb = "/home/rajroy/result_dump/test/temp_test/6VYV_PLH.pdb"
# _outputs_dir = "/home/rajroy/result_dump/test/temp_test/"


def main(_input_struct_pdb,_outputs_dir):
    antigen_chain = os.path.basename(_input_struct_pdb).split("_")[1].split(".")[0][-1]
    antibody_chains = os.path.basename(_input_struct_pdb).split("_")[1].split(".")[0][0:-1]

    input_struct_dir = _input_struct_pdb
    outputs_dir = _outputs_dir

    output_new_antigen_fasta_dir = outputs_dir + "antigen_fasta/"
    output_new_antibody_fasta_dir = outputs_dir + "antibody_fasta/"
    output_new_label_dir = outputs_dir + "labels/"
    antigen_structure_dir = outputs_dir + "antigen_structure/"
    antibody_structure_dir = outputs_dir + "antibody_structure/"

    os.makedirs(output_new_antigen_fasta_dir, exist_ok=True)
    os.makedirs(output_new_antibody_fasta_dir, exist_ok=True)
    os.makedirs(output_new_label_dir, exist_ok=True)
    os.makedirs(antigen_structure_dir, exist_ok=True)
    os.makedirs(antibody_structure_dir, exist_ok=True)

    # 6VYV_LP_H_interacting_residues_distance
    combined_epitope_array = []
    name = os.path.basename(_input_struct_pdb).split("_")[0]
    complex_structure_file_name = input_struct_dir
    # for sabdab
    combnined_epitope_array = []
    if os.path.exists(complex_structure_file_name):
        ###Remore Residue without CA
        complex_file = clean_utils.contents_to_info(clean_utils.read_pdb(complex_structure_file_name))
        complex_file = list(filter(lambda x: (x.alt_loc != 'B'), copy.deepcopy(complex_file)))
        complex_file = clean_utils.remove_duplicate_ca(complex_file)
        complex_file = clean_utils.remove_residues_without_CA(complex_file)
        antigen_pdb = clean_utils.separate_by_chain(complex_file, antigen_chain)
        anitgen_fasta_file_name = output_new_antigen_fasta_dir + name + "_" + antigen_chain + ".fasta"

        for antibody_chain in antibody_chains:
            antibody_pdb = clean_utils.separate_by_chain(complex_file, antibody_chain)
            antigen_serial_fix_a = clean_utils.fix_res_num_atom(antigen_pdb)
            antibody_serial_fix_a = clean_utils.fix_res_num_atom(antibody_pdb)
            clean_utils.convert_to_pdb(antibody_serial_fix_a,
                                       antibody_structure_dir + name + "_" + antibody_chain + ".pdb")
            clean_utils.convert_to_pdb(antigen_serial_fix_a,
                                       antigen_structure_dir + name + "_" + antigen_chain + ".pdb")
            anitbody_fasta_file_name = output_new_antibody_fasta_dir + name + "_" + antibody_chain + ".fasta"
            antibody_fasta = clean_utils.get_fasta_from_pdb_array(complex_file, antibody_chain)
            antibody_fasta_str = clean_utils.get_fasta_format(name + "_" + antibody_chain, antibody_fasta)
            clean_utils.write2File(anitbody_fasta_file_name, antibody_fasta_str)
            # ensuring same fasta  for antigen
            temp_antigen_pdb = clean_utils.contents_to_info(
                clean_utils.read_pdb(antigen_structure_dir + name + "_" + antigen_chain + ".pdb"))
            antigen_fasta = clean_utils.get_fasta_from_pdb_array(temp_antigen_pdb, antigen_chain)
            fasta_str = clean_utils.get_fasta_format(name + "_" + antigen_chain, antigen_fasta)
            clean_utils.write2File(anitgen_fasta_file_name, fasta_str)
            # ensuring same fasta  for antibody
            temp_antibody_pdb = clean_utils.contents_to_info(
                clean_utils.read_pdb(antibody_structure_dir + name + "_" + antibody_chain + ".pdb", ))
            antibody_fasta = clean_utils.get_fasta_from_pdb_array(temp_antibody_pdb, antibody_chain)
            antibody_fasta_str = clean_utils.get_fasta_format(name + "_" + antibody_chain, antibody_fasta)
            clean_utils.write2File(anitbody_fasta_file_name, antibody_fasta_str)
            dist, contact = clean_utils.get_distance_map_heavy_atom(antibody_serial_fix_a, antigen_serial_fix_a)
            dist_file_name = output_new_label_dir + name + "_dist_" + antigen_chain + antibody_chain + ".txt"
            contact_file_name = output_new_label_dir + name + "_contact_6_" + antigen_chain + antibody_chain + ".txt"
            HRF_file_name = output_new_label_dir + name + "_HRF_6_" + antigen_chain + antibody_chain + ".txt"
            np.savetxt(dist_file_name, dist, fmt="%.3f")
            np.savetxt(contact_file_name, contact, fmt="%.3f")
            ## contacts and distance in term of interace
            HRF_data_str = ""
            nz = np.argwhere(dist <= 6)
            # save contacts in HRF
            for r, c in nz:
                HRF_data_str += "{0},{1},{2}".format(r + 1, c + 1, dist[r, c]) + "\n"
                # print((r + 1, c + 1, dist[r, c]))
                combined_epitope_array.append(c + 1)
            clean_utils.write2File(HRF_file_name, HRF_data_str)
        combined_labels_name = output_new_label_dir + "{0}_{1}_{2}_interacting_residues_distance.txt".format(name,
                                                                                                             antibody_chains,
                                                                                                             antigen_chain)
        combined_epitope_array = np.unique(combined_epitope_array)
        combined_epitope_str = clean_utils.get_str_from_array(combined_epitope_array)
        clean_utils.write2File(combined_labels_name, combined_epitope_str)


# main()



