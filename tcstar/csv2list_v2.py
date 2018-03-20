import os
import pandas as pd
import numpy as np
import random

path = 'C:/Users/Oriol Barbany/Google Drive/UPC/4B/TFG/src/Multi-speaker-Neural-Vocoder/tcstar/'

files = []
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path, i)) and 'spk' in i:
        files.append(i)

totalLen = 0
IDList = np.empty((len(files),), dtype=object)
for file in files:
    table = pd.read_csv(path + file)
    a = files.index(file)
    IDList[files.index(file)] = table.index.tolist()
    totalLen += len(IDList[files.index(file)])

# Randomly take audio files from corpus 11 and 33 of each of the speakers
open('wav.list', 'w').close()           # Clear file before appending
generalFile = open(path + 'wav.list', "a")
partitions = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
for part in partitions:
    open('wav_' + part + '.list', 'w').close()           # Clear file before appending
    outFile = open(path + 'wav_' + part + '.list', "a")
    lenPart = totalLen*partitions[part]
    for i in range(1, round(lenPart)):
        file = random.choice(files)
        index = files.index(file)
        ID = random.choice(IDList[index])
        table = pd.read_csv(path + file)
        outFile.write(table['File Name'][ID][3:5] + '/' + table['File Name'][ID] + '\n')
        generalFile.write(table['File Name'][ID][3:5] + '/' + table['File Name'][ID] + '\n')
        IDList[index].remove(ID)
        if len(IDList[index]) == 0:
            IDList = np.delete(IDList, index, 0)
            files.remove(file)