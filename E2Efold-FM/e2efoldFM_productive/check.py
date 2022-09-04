import os

import requests
import zipfile

# api-endpoint


# location given here

# defining a params dict for the parameters to be sent to the API
URL = "https://proj.cse.cuhk.edu.hk/rnafm/api/predict_all"
def convert_to_fasta(fileName):
    fileSeq = open(fileName, 'r')
    with open(fileName+'.fasta', 'w') as fileF:
        fileF.write('>' + fileName + '\n')
        fileF.write(fileSeq.read())


def get_rna_ss(filename):
    file = {'predict_all_file': open(filename+'.fasta', 'rb')}
    params = {'date_id': '1'}
    # sending get request and saving the response as response object
    r = requests.post(url=URL, files=file, params=params)
    res = requests.get(url=URL, params={'filename': filename+'.fasta', 'date_id': '1'})

    with open(filename+'.zip', 'wb') as outFile:
        outFile.write(res.content)
    with zipfile.ZipFile(filename+".zip", "r") as zip_ref:
        zip_ref.extractall(filename.split('.')[0])
    os.remove(filename+'.zip')
    os.remove(filename+'.fasta')


convert_to_fasta('AB184819.seq')
get_rna_ss('AB184819.seq')
