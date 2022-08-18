import numpy as np
import os

file_path = os.path.join("/mnt","c","Users","mclap","OneDrive","Documents","vchips","code","predict_results","output.npz")

data = np.load(file_path)

distmat = data['distmat']

print(distmat.shape)