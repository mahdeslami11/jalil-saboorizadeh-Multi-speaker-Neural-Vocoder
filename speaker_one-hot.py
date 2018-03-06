path = 'C:/Users/obarbany/Desktop/python/'

searchfile = open(path + "lof.txt", "r")

outfile = open(path + "lof_wav.txt", "w")

spk = set()

# Search for unique speakers in .wav files

for line in searchfile:

    if ".wav" in line:

        outfile.write(line)

        spk_line = line[5:8]

        if spk_line not in spk:
            spk.add(spk_line)

searchfile.close()

outfile.close()

# Dictionary of one hot encode

index = dict((s, i) for i, s in enumerate(spk))

for key, value in index.items():
    one_hot = [0 for _ in range(len(index.items()))]

    one_hot[value] = 1

    index[key] = one_hot

searchfile = open(path + "lof_wav.txt", "r")

outfile = open(path + "spk.csv", "w")

# Substitute speaker ID with one-hot encoding

for line in searchfile:
    outfile.write(line.rstrip() + ';' + str(index[line[5:8]]) + '\n')

searchfile.close()

outfile.close()