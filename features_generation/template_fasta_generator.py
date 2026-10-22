import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs import text_file_reader
from utils import get_aligned_sequences

def uppercase_indexes(s):
    return [i for i, ch in enumerate(s) if ch.isupper()]

template_label ="/home/rajroy/result_dump/test/temp_test/label_fasta/6VYV_H_label.txt"
target_file ="/home/rajroy/result_dump/test/temp_test/6W1C_H.fasta"

template_label_fasta = text_file_reader(template_label)[1]
target_fasta = (text_file_reader(target_file)[1]).lower()

# print(template_label_fasta)
# print(target_fasta)

aln_repr, aln_temp = get_aligned_sequences(target_fasta, template_label_fasta)
# print(aln_repr)
# print(aln_temp)
template_label = uppercase_indexes(aln_temp)
index  = 0
str_fasta_new = ""
for char in aln_repr:
    if index in template_label:
        str_fasta_new = str_fasta_new+  char.upper()
    else:
        str_fasta_new = str_fasta_new + char

    index += 1
# print(str_fasta_new)
str_fasta_new = str_fasta_new.replace("-","")
str_fasta_new = str_fasta_new.replace("x","")

print("final")
print(str_fasta_new)
print(uppercase_indexes(str_fasta_new))