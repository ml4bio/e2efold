import os
import pandas as pd

ids = {}
lengths = {}


with open("/share/liyu/RNA3D/shen_ss_eval/blast/result.xls", "r") as f:
    lines = f.readlines()

    record_need_length = False

    for index, line in enumerate(lines):
        if ">" in line:
            record_name = line.strip().replace(">", "")
            if record_name not in ids:
                ids[record_name] = []
            record_need_length = True

        if record_need_length == True and "Length" in line:
            lengths[record_name] = int(line.strip().split("=")[1])
            record_need_length = False

        if "Identities" in line:
            num, identity = line.split(")")[0].split("(")
            num = int(num.split("=")[-1].split("/")[-1])
            ids[record_name].append([identity, num])

        print("{}/{}".format(index, len(lines)))


records = []
for key in ids.keys():
    print(ids[key], lengths[key])
    max_id = 0
    for id, num in ids[key]:
        id_value = int(id.replace("%", ""))
        id_value = id_value * num / lengths[key]
        if id_value > max_id:
            max_id = id_value
    if max_id < 80:
        records.append([key, max_id, 0])
    else:
        records.append([key, max_id, 1])
    print(len(records))

df = pd.DataFrame(records, columns=["filename", "identity", ">=80"])
df.to_csv("/share/liyu/RNA3D/shen_ss_eval/blast/dentity.csv")