import os
import numpy as np
import pandas as pd 
import argparse
import csv
from tqdm import tqdm


def ensemble(opt):

    test = pd.DataFrame()
    test['image_id'] = sorted(list(os.listdir(opt.test_dir)))

    model_preds = []
    for npy_file in os.listdir(opt.npy_folder):
        npy_path = os.path.join(opt.npy_folder, npy_file)
        result = np.load(npy_path)
        model_preds += [result]
    
    avg_tst_preds = np.mean(model_preds, axis=0)
    
    test['label'] = np.argmax(avg_tst_preds, axis=1)
    test['prob'] = np.amax(avg_tst_preds, axis=1)

    if opt.pseudo_label:
        field_names = ['image_id', 'label']
        pseudo_labeling = []
        for i, row in tqdm(test.iterrows()):
            if row["prob"] >= opt.pseudo_thr:
                tmp = {
                    'image_id': row["image_id"],
                    'label': row["label"]
                }
                pseudo_labeling.append(tmp)

        with open(os.path.join(opt.save_dir, 'pseudo_labeling.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = field_names)
            writer.writeheader()
            writer.writerows(pseudo_labeling)

    test[['image_id', 'label']].to_csv(os.path.join(opt.save_dir, 'vit_submission.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, help='test directory')
    parser.add_argument('--npy_folder', type=str, help='npy directory')
    parser.add_argument('--pseudo_label', action='store_true', help='activate pseudo-labeling')
    parser.add_argument('--pseudo_thr', type=int, default=0.99, help='threshold to get pseudo-labeling')
    parser.add_argument('--save_dir', type=str, default="/home/datdoc/win-home/Desktop/ML/icdar2021")

    opt = parser.parse_args()  
    ensemble(opt)