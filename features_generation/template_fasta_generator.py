import os.path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs import text_file_reader, write2File
from utils import get_aligned_sequences, get_fasta_format


def uppercase_indexes(s):
    return [i for i, ch in enumerate(s) if ch.isupper()]

# template_label ="/home/rajroy/result_dump/test/temp_test/label_fasta/6VYV_H_label.txt"
# target_file ="/home/rajroy/result_dump/test/temp_test/6W1C_H.fasta"


def main(template_label_file,target_file,_output_dir):
    template_label_fasta = text_file_reader(template_label_file)[1]
    target_fasta = (text_file_reader(target_file)[1]).lower()
    aln_repr, aln_temp = get_aligned_sequences(target_fasta, template_label_fasta)
    template_label = uppercase_indexes(aln_temp)
    index = 0
    str_fasta_new = ""
    for char in aln_repr:
        if index in template_label:
            str_fasta_new = str_fasta_new + char.upper()
        else:
            str_fasta_new = str_fasta_new + char

        index += 1
    str_fasta_new = str_fasta_new.replace("-", "")
    str_fasta_new = str_fasta_new.replace("x", "")
    # print("final")
    # print(str_fasta_new)
    # print(uppercase_indexes(str_fasta_new))
    name = "temp_{0}_for_{1}".format(os.path.basename(template_label_file),
                                                                  os.path.basename(target_file))
    _output_file = _output_dir + "/{0}.fasta".format(name)

    final_content = get_fasta_format(name,str_fasta_new)

    write2File(_output_file,final_content)

    return _output_file