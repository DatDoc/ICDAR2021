# SPLIT GROUPKFOLD

import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
import pandas as pd 
import os 
from tqdm import tqdm


hand_folder = "/home/datdoc/win-home/Desktop/ML/icdar2021/code/handwritten.csv"
printed_folder = "/home/datdoc/win-home/Desktop/ML/icdar2021/code/printed.csv"
pseudo_label_folder = "/home/datdoc/win-home/Desktop/ML/icdar2021/pseudo_labeling.csv"

train_df = pd.read_csv(pseudo_label_folder)

dst_path = "/home/datdoc/win-home/Desktop/ML/icdar2021/repo/ICDAR2021/split"

skf  = StratifiedKFold(n_splits = 5, shuffle=True)

train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(X=train_df.image_id, y=train_df["label"])):
    train_df.loc[val_idx, 'fold'] = fold

train_df.to_csv(os.path.join(dst_path, "skf_pseudo_data.csv"), index=False)

    
    
