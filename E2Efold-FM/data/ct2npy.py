# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com

Convert the ct ((Connectivity Table)) file of RNA secondary structure to npy file (regarding as Connectivity Map, "cm")

python /data/chenjiayang/projects/ESM-GeneralSeq/redevelop/tools/ct2npy.py --ct_dir=xx --cm_dir=xx
"""

import os
import pandas as pd
import numpy as np
import argparse

def ct2npy(ct_file, cm_dir):
    with open(ct_file, "r") as f:
        first_line = f.readline()
        if ">seq length:" in first_line:
            #print(first_line.split("\t")[0].replace(">seq length:", ""))
            seq_len = int(first_line.split("\t")[0].replace(">seq length: ", ""))
        else:
            seq_len = int(first_line.strip(" ").split(" ")[0].split("\t")[0])
        print(seq_len)

    df = pd.read_csv(ct_file, sep='\s+', skiprows=1, header=None)
    x = df.loc[:, 0].values
    y = df.loc[:, 4].values

    #seq_len = x.shape[0]
    contact = np.zeros([seq_len, seq_len]).astype(np.int8)
    for pair in zip(x, y):
        if pair[1] <= 0:
            continue
        contact[pair[0]-1, pair[1]-1] = 1

    if (contact ==contact.T).all() != True:
        raise Exception("asymmetric contact")

    # save
    _, ct_filename = os.path.split(ct_file)
    prename, ext = os.path.splitext(ct_filename)
    if os.path.exists(cm_dir) != True:
        os.makedirs(cm_dir)
    cm_file = os.path.join(cm_dir, prename)

    np.save(cm_file, contact)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_dir', type=str, default=None)
    parser.add_argument('--cm_dir', type=str, default=None)
    args = parser.parse_args()

    ct_dir = args.ct_dir
    cm_dir = args.cm_dir

    dot_filenames = os.listdir(ct_dir)
    num_files = len(dot_filenames)
    for index, dot_filename in enumerate(dot_filenames):
        prename, ext = os.path.splitext(dot_filename)
        if ext != ".ct":
            continue
        ct2npy(os.path.join(ct_dir, dot_filename), cm_dir)
        print("complete {}/{}".format(index+1, num_files))