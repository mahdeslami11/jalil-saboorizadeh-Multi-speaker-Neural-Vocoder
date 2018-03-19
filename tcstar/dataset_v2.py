import os
import pandas as pd
import random

path = 'C:/Users/Oriol Barbany/Google Drive/UPC/4B/TFG/src/Multi-speaker-Neural-Vocoder/tcstar/'

files = []
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path, i)) and 'spk' in i:
        files.append(i)

# Randomly take audio files from corpus 11 and 33 of each of the speakers
open('wav.list', 'w').close()           # Clear file before appending
generalFile = open(path + 'wav.list', "a")
partitions = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
for part in partitions:
    open('wav_' + part + '.list', 'w').close()           # Clear file before appending
    outFile = open(path + 'wav_' + part + '.list', "a")
    minTimePart = minTime*partitions[part]
    for file in files:
        time = 0
        table = pd.read_csv(path + file)
        IDList = table.index.tolist()
        while time < minTimePart and IDList:
            ID = random.choice(IDList)
            outFile.write(table['File Name'][ID][3:5] + '/' + table['File Name'][ID] + '\n')
            generalFile.write(table['File Name'][ID][3:5] + '/' + table['File Name'][ID] + '\n')
            time += table['Duration (s)'][ID]
            IDList.remove(ID)