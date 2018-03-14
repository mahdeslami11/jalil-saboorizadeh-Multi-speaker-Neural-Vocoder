import os
import pandas as pd

path = 'C:/Users/Oriol Barbany/Google Drive/UPC/4B/TFG/src/Multi-speaker-Neural-Vocoder/tcstar/'

# Search files in directory starting with lof (list of files)
files = []
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path, i)) and 'lof' in i:
        files.append(i)

for file in files:
    lofFile = open(path + file, "r")
    fileNames = lofFile.read().splitlines()
    aux = file.split("_")[1]
    durFile = open(path + "duration_" + aux, "r")
    durations = durFile.read().splitlines()
    data = {'File Name': fileNames, 'Duration (s)': durations}
    df = pd.DataFrame(data, columns=['File Name', 'Duration (s)'])
    df.to_csv('spk' + file.split("_")[1].split(".")[0] + '.csv')