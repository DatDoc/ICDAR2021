import argparse
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

from engine import *
from utils import *
from models import Classifier

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
import torch.nn as nn

categ = {
    "Arabic": 0,
    "Bangla": 1,
    "Gujrati": 2,
    "Gurmukhi": 3,
    "Hindi": 4,
    "Japanese": 5,
    "Kannada": 6, 
    "Malayalam": 7,
    "Oriya": 8,
    "Roman": 9, 
    "Tamil": 10,
    "Telugu": 11,
    "Thai": 12
}

def run_training(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    work_dir, epochs, train_batch, valid_batch, weights = \
        opt.work_dir, opt.epochs, opt.train_bs, opt.valid_bs, opt.weights
        
    # Directories
    last = os.path.join(work_dir, 'last.pt')
    best = os.path.join(work_dir, 'best.pt')

    # --------------------------------------
    # Setup train and validation set
    # --------------------------------------
    data = pd.read_csv(opt.train_csv)
    images_path = opt.data_dir

    n_classes = 13 # fixed coding :V

    # data['label'] = data.apply(lambda row: categ[row["label"]], axis =1)

    train_loader, val_loader = prepare_dataloader(
        data, opt.fold, train_batch, valid_batch, opt.img_size, opt.num_workers, data_root=images_path)
    
    if not opt.ovr_val:
        handwritten_data = pd.read_csv(opt.handwritten_csv)
        printed_data = pd.read_csv(opt.printed_csv)
        handwritten_data['label'] = handwritten_data.apply(lambda row: categ[row["label"]], axis =1)
        printed_data['label'] = printed_data.apply(lambda row: categ[row["label"]], axis =1)
        _, handwritten_val_loader = prepare_dataloader(
            handwritten_data, opt.fold, train_batch, valid_batch, opt.img_size, opt.num_workers, data_root=images_path)

        _, printed_val_loader = prepare_dataloader(
            printed_data, opt.fold, train_batch, valid_batch, opt.img_size, opt.num_workers, data_root=images_path)
    
    # --------------------------------------
    # Models
    # --------------------------------------
  
    model = Classifier(model_name=opt.model_name, n_classes=n_classes, pretrained=True).to(device)

    if opt.weights is not None:
        cp = torch.load(opt.weights)
        model.load_state_dict(cp['model'])
    
    # -------------------------------------------
    # Setup optimizer, scheduler, criterion loss
    # -------------------------------------------

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    scaler = GradScaler()

    loss_tr = nn.CrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # --------------------------------------
    # Setup training 
    # --------------------------------------
    if os.path.exists(work_dir) == False:
        os.mkdir(work_dir)

    best_loss = 1e5
    start_epoch = 0
    best_epoch = 0 # for early stopping

    if opt.resume == True:
        checkpoint = torch.load(last)

        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_loss = checkpoint["best_loss"]


    # --------------------------------------
    # Start training
    # --------------------------------------
    print("[INFO] Start training...")
    for epoch in range(start_epoch, epochs):
        train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, scaler=scaler)
        with torch.no_grad():
            if opt.ovr_val:
                val_loss = valid_one_epoch_overall(epoch, model, loss_fn, val_loader, device, scheduler=None)
            else:
                val_loss = valid_one_epoch(epoch, model, loss_fn, handwritten_val_loader, printed_val_loader, device, scheduler=None)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_loss': best_loss
                },
                os.path.join(best))
        
                print('best model found for epoch {}'.format(epoch+1))

        torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss
            },
            os.path.join(last))

        if epoch - best_epoch > opt.patience:
            print("Early stop achieved at", epoch+1)
            break
            
            
    del model, optimizer, train_loader, val_loader, scheduler, scaler
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/content/train_images', help='image directory')
    parser.add_argument('--train_csv', type=str,default='/content/Shopee/split/group5folds.csv', help='group5folds')
    parser.add_argument('--handwritten_csv', type=str, help='path to skf_handwritten.csv')
    parser.add_argument('--printed_csv', type=str, help='path to skf_printed.csv')
    parser.add_argument('--fold', type=int, help='fold number')
    parser.add_argument('--seed', type=int, default=2021, help='for reproduce')
    parser.add_argument('--epochs', type=int, default=20, help='number of epoch')
    parser.add_argument('--train_bs', type=int, default=16, help='train batch size')
    parser.add_argument('--valid_bs', type=int, default=16, help='validation batch size')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--work_dir', type=str, default='CLIP/fold0', help='path to save model')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--weights', type=str, help='add weight path')
    parser.add_argument('--model_name', type=str, default='tf_efficientnet_b4', help="ViT-B-32, RN50, RN50x4, RN101")
    parser.add_argument('--patience', type=int, default=5, help="set early stopping patience")
    parser.add_argument('--img_size', type=int, default=384, help='resize the image')
    parser.add_argument('--ovr_val', action='store_true', help='overall validation')
    opt = parser.parse_args()

    seed_torch(seed=opt.seed)
    run_training(opt)