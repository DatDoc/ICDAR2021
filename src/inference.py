import argparse
from tqdm import tqdm
import pandas as pd

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


def run_infer(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_names = ['tf_efficientnet_b4_ns','resnext50_32x4d','vit_base_patch16_384']
    n_classes = 13
    model_weights = [1,1,1,1,1,1]
    weight_path = opt.weight_path
    weights =  sorted(os.listdir(weight_path))
    
    for ix, model_arch in enumerate(model_names):

        TEST_DIR = opt.test_dir
        test = pd.DataFrame()
        test['image_id'] = list(os.listdir(TEST_DIR))

        if model_arch=='vit_base_patch16_384':
            testset= ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms_vit())
        else: 
            testset= ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(opt.img_size))

        tst_loader = DataLoader(
            testset, 
            batch_size=opt.valid_bs,
            num_workers=opt.num_workers,
            shuffle=False,
            pin_memory=False)
        

        model = Classifier(model_arch, n_classes).to(device)

        tst_preds = []

        for i,weight in enumerate(weights[ix*2:ix*2+2]):    

            model.load_state_dict(torch.load(os.path.join(weight_path, weight))['model'])

            with torch.no_grad():
                for _ in range(CFG['tta']):
                    tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta'] * inference_one_epoch(model, tst_loader, device)]
        avg_tst_preds = np.mean(tst_preds, axis=0)

        if not (os.path.isdir('./total_preds')): os.mkdir('./total_preds')
        np.save('./total_preds/total_preds.npy', tst_preds)

        if not (os.path.isdir('./mean_preds')): os.mkdir('./mean_preds')
        np.save('./mean_preds/mean_preds.npy', avg_tst_preds)

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
    parser.add_argument('--weight_path', type=str, help='path to stored weights')
    parser.add_argument('--img_size', type=int, default=224, help='resize the image')
    parser.add_argument('--tta', type=int, default=2, help='number of tta')
    opt = parser.parse_args()

    seed_torch(seed=opt.seed)
    run_infer(opt)