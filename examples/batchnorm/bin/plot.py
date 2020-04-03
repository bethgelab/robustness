import glob
import os.path as osp
import re
import numpy as np
from multiprocessing import Pool
import pandas as pd

regex = re.compile(r'(\w*)_([\w_]*)-([0-9]*)_bsz([0-9]*)\.npz')

def load_fname(fname):
    
    keys = [("model",str), ("corruption", str), ("severity", int), ("batchsize", int)]
    (vals), = regex.findall(osp.basename(fname))
    args = {k:func(v) for (k,func),v in zip(keys,vals)}
    
    arr = np.load(fname, allow_pickle = True)
    keys = ['top1', 'top5', 'loss']
    
    return dict(**args, **{k:float(arr[k]) for k in keys})
    
fnames = glob.glob("/home/ubuntu/local/emissions/*.npz")
with Pool() as p: results = p.map(load_fname, fnames)
results = pd.DataFrame(results)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

for k in [1,5]:

    sns.set_context("poster")
    plt.figure(figsize=(10,10))
    sns.lineplot(data = results.groupby(["corruption", "batchsize"]).mean().reset_index(),
                 x = "batchsize", y = f"top{k}", hue = "corruption") #, style = "severity")
    plt.xscale("log")
    plt.title(f"Top-{k} performance, averaged over severities")
    plt.legend(loc=(1,0))
    sns.despine()
    plt.show()

    sns.set_context("poster")
    plt.figure(figsize=(10,10))
    sns.lineplot(data = results.groupby(["severity", "batchsize"]).mean().reset_index(),
                 x = "batchsize", y = f"top{k}", hue = "severity") #, style = "severity")
    plt.xscale("log")
    plt.title(f"Top-{k} performance, averaged over corruptions")
    plt.legend(loc=(1,0))
    sns.despine()
    plt.show()
