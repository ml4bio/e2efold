import os
import shutil
import pandas as pd

# ArchiveII
"""
root_dir = "/share/liyu/RNA/Data/E2Efold-SS/preprocessed/archiveII/"
dest_dir = "/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_seqs/"
ann_filename = "all_600.csv"
#"""

# PDB
"""
root_dir = "/share/liyu/RNA/Data/SPOT-RNA/preprocessed/PDB/"
dest_dir = "/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_seqs/"
ann_dir = os.path.join(root_dir, "ann")
ann_filename = "test2.csv"   # test2
#"""

# RNAStralign
"""
root_dir = "/share/liyu/RNA/Data/E2Efold-SS/preprocessed/rnastralign/"
dest_dir = "/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_seqs/"
ann_dir = os.path.join(root_dir, "ann")
ann_filename = "test_no_redundant_600.csv"   #filename.replace("_p10", "").replace("_p1", "")
#"""

# bpRNA (1305)
#"""
root_dir = "/share/liyu/RNA/Data/SPOT-RNA/preprocessed/bpRNA/"
dest_dir = "/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/short_seqs/"
ann_dir = os.path.join(root_dir, "ann")
ann_filename = "test.csv"   #filename.replace("_p10", "").replace("_p1", "")
#"""



if os.path.exists(dest_dir) != True:
    os.makedirs(dest_dir)

seq_dir = os.path.join(root_dir, "seq")
ref_file = os.path.join(ann_dir, ann_filename)

ref_df = pd.read_csv(ref_file)

filenames = ref_df["filename"].values

lengths = ref_df["length"].values

#for index, row in ref_df.iterrows():
#    filename


count = 0
for index, filename in enumerate(filenames):
    length = lengths[index]
    if length > 500:
        continue

    count += 1

    src_seq_file = os.path.join(seq_dir, filename.replace("_p10", "").replace("_p1", "")+".seq")

    dest_seq_file = os.path.join(dest_dir, filename.replace("_p10", "").replace("_p1", "")+".fasta")

    print("{}/{} {} {}".format(index, len(filenames), filename, length))


    shutil.copy2(src_seq_file, dest_seq_file)
print(count)