#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import re
import sys


show_pp = True
show_pp = False

def PP(x, compute_pp=False):
    if compute_pp:
        return np.exp2(x)
    else:
        return x


def plotfigure(fname):
    print('Plot ', fname)
    f = open(fname)
    iterpat = re.compile('training_loss:.*time:')
    trainpat = re.compile('training_loss: ([-0-9.]+)')
    valpat   = re.compile('validation_loss: ([-0-9.]+)')
    testpat  = re.compile('test_loss: ([-0-9.]+)')

    # Count data
    ntr = nte = nva = 0
    for line in f:
        if trainpat.search(line):
            ntr += 1
        if valpat.search(line):
            nva += 1
        if testpat.search(line):
            nte += 1


    idxtr = np.zeros(ntr, dtype=int)
    idxva = np.zeros(nva, dtype=int)
    idxte = np.zeros(nte, dtype=int)

    dattr = np.zeros(ntr, dtype=float)
    datva = np.zeros(nva, dtype=float)
    datte = np.zeros(nte, dtype=float)

    # rewind file, read again, saving values
    f.seek(0)
    ntr = nte = nva = 0
    n = 0
    for line in f:
        if iterpat.search(line):
            n +=1

        m = trainpat.search(line);
        if m:
            idxtr[ntr] = n
            dattr[ntr] = float(m.group(1))
            ntr += 1

        m = valpat.search(line);
        if m:
            idxva[nva] = n
            datva[nva] = float(m.group(1))
            nva += 1
        
        m = testpat.search(line);
        if m:
            idxte[nte] = n
            datte[nte] = float(m.group(1))
            nte += 1
        
    
    nepochs  = len(datva)
    niters   = len(dattr)
    it_per_e = niters//nepochs

    print('epochs=', len(datva), len(datte))
    print('iterations=', len(dattr))
    print('iterations/batch=', len(dattr)//len(datte))

    fig = plt.figure()
    plt.xlabel('iteration')
    plt.grid()
    
    ylimits = PP(np.array([1.,4.]),show_pp)
    plt.ylim(ylimits)
    plt.plot(idxtr, PP(dattr,show_pp), 'b',
             idxva, PP(datva,show_pp), 'g',
             idxte, PP(datte,show_pp), 'r--')
    plt.gca().set_yticks(np.arange(ylimits[0], ylimits[1], (ylimits[1]-ylimits[0])/10)) #0.25))
    plt.gca().set_xticks(np.arange(1, niters, it_per_e*5))

    #ax = fig.gca()
    #ax.set_yticks(np.arange(0, 9., 0.5))

    if show_pp:
        plt.ylabel('PP (Perplexity)')
        plt.legend(['train PP', 'valid PP', ' test PP'])
    else:
        plt.ylabel('NLL (negative log. likelihood)')
        plt.legend(['train NLL', 'valid NLL', ' test NLL'])
    
    plt.title('({2}) epochs={0}; iterations/epoch={1}'.format(len(datte),
                                                                  len(dattr)//len(datte),
                                                                  fname,))
    plt.savefig(fname + '.png')



if len(sys.argv) == 1:
    fnames = ['log']
else:
    fnames = sys.argv[1:]

    
for fname in fnames:
    plotfigure(fname)
plt.show()

    
