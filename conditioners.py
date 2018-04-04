import numpy as np
from interpolate import interpolation


def cond_max_min(datasets_path, cond_path, cond_glob, partition, npy_name_max_cond, npy_name_min_cond):
    print('Computing conditioning extrema')
    # Get last file of cond_glob
    file_names = open(datasets_path + 'wav_' + partition + '.list', 'r').read().splitlines()
    from_file = len(file_names)
    # Get file names from wav list
    file_names = open(datasets_path + 'wav.list', 'r').read().splitlines()
    file_names = file_names[from_file:]
    # Load each of the files from the list. Note that extension has to be added
    for file in file_names:
        print('Analyzing file ' + file)
        # Load CC conditioner
        c = np.loadtxt(cond_path + file + '.cc')
        c = c.reshape(-1, c.shape[1])
        (num_ceps, _) = c.shape

        # Load LF0 conditioner
        f0file = np.loadtxt(cond_path + file + '.lf0')
        f0, _ = interpolation(f0file, -10000000000)
        f0 = f0.reshape(f0.shape[0], 1)

        # Load FV conditioner
        fvfile = np.loadtxt(cond_path + file + '.gv')
        fv, uv = interpolation(fvfile, 1e3)
        num_fv = fv.shape[0]
        uv = uv.reshape(num_fv, 1)
        fv = fv.reshape(num_fv, 1)

        # Concatenate all speech conditioners
        cond = np.concatenate((c, f0), axis=1)
        cond = np.concatenate((cond, fv), axis=1)
        cond = np.concatenate((cond, uv), axis=1)

        # Append/Concatenate current speech conditioners
        cond_glob = np.concatenate((cond_glob, cond), axis=0)

    # Compute maximum and minimum for each partition
    max_cond = np.amax(cond_glob, axis=0)
    min_cond = np.amin(cond_glob, axis=0)

    # Save maximum and minimum conditioners to replicate normalization when generating
    np.save(npy_name_max_cond, max_cond)
    np.save(npy_name_min_cond, min_cond)
