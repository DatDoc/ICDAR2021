import os
import numpy as np
import pandas as pd 
import argparse


def ensemble(opt):

    model_preds = []
    for npy_file in os.listdir(opt.npy_folder):
        npy_path = os.path.join(opt.npy_folder, npy_file)
        result = np.load(npy_path)
        model_preds += [result]

    avg_tst_preds = np.mean(model_preds, axis=0)

    test['label'] = np.argmax(avg_tst_preds, axis=1)
    test.to_csv('submission.csv', index=False)
    test.head()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_folder', type=str, help='npy directory')
    parser.add_argument('--pseudo_thr', type=int, default=0.9, help='threshold to get pseudo-labeling')
    
    opt = parser.parse_args()  
    ensemble(opt)