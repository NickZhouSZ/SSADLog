import random

import pandas as pd
import numpy as np
import re
import string

import time
from datetime import datetime
import collections

log_file = "./data/split/BGL_train80.log"
#log_file = "./data/split/Spirit1G_train80.log"
#log_file = "./data/split/Thunderbird10M_train80.log"
output_log_file = "./data/split/BGL_train80_reduced_normal_60.log"
#output_log_file = "./data/split/Spirit1G_train80_reduced_normal_60.log"
#output_log_file = "./data/split/Thunderbird10M_train80_reduced_normal_60.log"

def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message

    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s

def remove_duplicates_normal(input_file, output_file, recent_lines):
    duplicated_line_count = 0
    out_tr = []
    out_ind = 0
    label = 0

    with open(input_file, mode="r", encoding='utf8',errors='ignore') as file:
        print("Loading", input_file)

        print("line 0:")
        first_line = file.readline() 
        print(first_line)
        # add first line
        cleanLine = clean(first_line).lower()
        recent_lines.append(cleanLine)
        out_tr.append(first_line)
        out_ind +=1

        for line_number, line in enumerate(file, start=1):
            if line.startswith('-'):
                label = 0
            else:
                label = 1

            cleanLine = clean(line).lower()
            if cleanLine not in recent_lines or label == 1:
                recent_lines.append(cleanLine)
                out_tr.append(line)
                out_ind +=1
                #print(f"Line {line_number}: {line}")
            else:
                duplicated_line_count +=1
                #print(f"dupicated Line {line_number}: {line}")
        print("Loaded", line_number+1, "lines!")
        print(f"Total Line {line_number+1}: output {out_ind} : duplicated {duplicated_line_count}")

    fo = open(output_file, "w")
    fo.writelines(out_tr)
    fo.close()
    print("write", len(out_tr), "lines!")

if __name__ == '__main__':
    recent_lines = collections.deque(maxlen=60)  # Store the last 60 lines
    remove_duplicates_normal(log_file, output_log_file, recent_lines)

