import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import torch 

from utils import *
from augmentations import *
from models.ViT import Classifier
from datasets import ICDARDataset

def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        image_preds = model(imgs)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def load_models(models_path, device):
    model_names = os.listdir(models_path)
    models = []
    n_class = 13
    for path in models_names:
        
        if "efficientnet" in path:
            model = Classifier("tf_efficientnet_b4_ns", n_classes).to(device)
        elif "vit" in path:
            model = Classifier("vit_base_patch16_384", n_classes).to(device)
        elif "resnext" in path:
            model = Classifier("resnext50_32x4d", n_classes).to(device)

        for idx in range(5): # number of fold
            model_path = os.path.join(path, "fold{}".format(idx))

            model.load_state_dict(torch.load(os.path.join(model_path, "best.pt"))['model'])

            models.append(model)
            del model

    return models


def run_infer(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch_weights = np.ones(15) # set weight for each arch base on f1 score on CV
   
    n_classes = 13
    weight_path = opt.weight_path
    tst_preds = []

    

    TEST_DIR = opt.test_dir
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir(TEST_DIR))

    if model_arch=='vit_base_patch16_384':
        testset= ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(384))
    else: 
        testset= ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(opt.img_size))

    tst_loader = DataLoader(
        testset, 
        batch_size=opt.valid_bs,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=False)

    models = load_models(weight_path, device)
    
    for model in models:

        with torch.no_grad():
            for _ in range(tta):
                tst_preds += [inference_one_epoch(model, tst_loader, device)]

    avg_tst_preds = np.mean(tst_preds, axis=0)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='/content/train_images', help='image directory')
    parser.add_argument('--test_csv', type=str,default='/content/Shopee/split/group5folds.csv', help='group5folds')
    parser.add_argument('--seed', type=int, default=2021, help='for reproduce')
    parser.add_argument('--valid_bs', type=int, default=16, help='validation batch size')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--work_dir', type=str, default='CLIP/fold0', help='path to save model')
    parser.add_argument('--weight_path', nargs='+', type=str, help='model.pt path(s)')  # /content/ViT
                                                                                        # /content/resnext
                                                                                        # /content/nfnet
    parser.add_argument('--img_size', type=int, default=224, help='resize the image')
    parser.add_argument('--tta', type=int, default=2, help='number of tta')
    opt = parser.parse_args()

    seed_torch(seed=opt.seed)
    run_infer(opt)