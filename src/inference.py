import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import torch 
from torch.utils.data import Dataset, DataLoader

from utils import *
from augmentations import *
from models import Classifier
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


def run_infer(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    n_classes = 13
    weight_path = opt.weight_path
    tst_preds = []

    TEST_DIR = opt.test_dir
    test = pd.DataFrame()
    test['image_id'] = sorted(list(os.listdir(TEST_DIR)))
    

    if "vit" in opt.model_arch:
        testset = ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(384))
    else:
        testset = ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(opt.img_size))
        
    
    tst_loader = DataLoader(
        testset, 
        batch_size=opt.valid_bs,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=False)

    
    print("[INFO] Found {} folds in weight path".format(len(os.listdir(weight_path))))
    print(os.listdir(weight_path))

    print("[INFO] Start inference ...")
    for fold in os.listdir(weight_path):
        print(fold)
        model = Classifier(opt.model_arch, n_classes).to(device)
        model_path = os.path.join(weight_path, fold, "best.pt")
        model.load_state_dict(torch.load(model_path)['model'])

        with torch.no_grad():
            for _ in range(opt.tta):
                tst_preds += [inference_one_epoch(model, tst_loader, device)]
        
        del model

    avg_tst_preds = np.mean(tst_preds, axis=0)
    if not (os.path.isdir(opt.work_dir)): 
        os.mkdir(opt.work_dir)
        
    np.save(os.path.join(opt.work_dir, "{}.npy".format(opt.model_arch)), avg_tst_preds)
    test['label'] = np.argmax(avg_tst_preds, axis=1)
    test.to_csv('submission.csv', index=False)
    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='/content/train_images', help='image directory')
    parser.add_argument('--model_arch', type=str, help='tf_efficientnet_b4_ns, vit_base_patch16_384, resnext50_32x4d')
    parser.add_argument('--seed', type=int, default=2021, help='for reproduce')
    parser.add_argument('--valid_bs', type=int, default=256, help='validation batch size')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--work_dir', type=str, default='CLIP/fold0', help='path to save model')
    parser.add_argument('--weight_path', type=str, help='model.pt path(s)')  # /content/ViT
                                                                                        # /content/resnext
                                                                                        # /content/nfnet
    parser.add_argument('--img_size', type=int, default=224, help='resize the image')
    parser.add_argument('--tta', type=int, default=1, help='number of tta')
    opt = parser.parse_args()

    seed_torch(seed=opt.seed)
    run_infer(opt)