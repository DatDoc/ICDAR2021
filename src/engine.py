import pandas as pd 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable

from datasets import ICDARDataset
from augmentations import *




def prepare_dataloader(df, fold, train_batch, valid_batch,img_sz,num_workers, data_root=TRAIN_IMAGES_DIR):

    trainset = df[df.fold != fold].reset_index(drop=True)
    validset = df[df.fold == fold].reset_index(drop=True)

    train_dataset = ICDARDataset(
        trainset, 
        image_root=data_root, 
        transforms=get_train_transforms(img_sz))

    valid_dataset = ICDARDataset(
        validset, 
        image_root=data_root, 
        transforms=get_valid_transforms(img_sz))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch, 
        pin_memory=False,
        drop_last=False,
        shuffle=True, 
        num_workers=num_workers)

    val_loader = DataLoader(
        valid_dataset, 
        batch_size=train_batch, 
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False)

    return train_loader, val_loader

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler):
    model.train()
    lst_out = []
    lst_label = []
    avg_loss = 0

    status = tqdm(enumerate(train_loader), total=len(train_loader), desc = "Training epoch " + str(epoch+1))
    for step, (images, labels) in status:
        images = images.to(device).float()
        labels = labels.to(device).long()
        with autocast():
            preds = model(images)
            lst_out += preds.argmax(1)
            lst_label += labels

            loss = loss_fn(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            avg_loss += loss.item() / len(train_loader)
            
    scheduler.step()
    accuracy = accuracy_score(y_pred=torch.tensor(lst_out), y_true=torch.tensor(lst_label))
    print('{} epoch - train loss : {}, train accuracy : {}'.\
          format(epoch + 1, np.round(avg_loss,6), np.round(accuracy*100,2)))

def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler):
    model.eval()
    lst_val_out = []
    lst_val_label = []
    avg_val_loss = 0
    status = tqdm(enumerate(val_loader), total=len(val_loader), desc = "Validating epoch " + str(epoch+1))
    for step, (images, labels) in status:
        val_images = images.to(device).float()
        val_labels = labels.to(device).long()

        val_preds = model(val_images)
        lst_val_out += val_preds.argmax(1)
        lst_val_label += val_labels
        loss = loss_fn(val_preds, val_labels)
                       
        avg_val_loss += loss.item() / len(val_loader)
    accuracy = accuracy_score(y_pred=torch.tensor(lst_val_out), y_true=torch.tensor(lst_val_label))
    print('{} epoch - valid loss : {}, valid accuracy : {}'.\
          format(epoch + 1, np.round(avg_val_loss, 6), np.round(accuracy*100,2)))
    return avg_val_loss