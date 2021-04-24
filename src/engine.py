import pandas as pd 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from datasets import ICDARDataset
from augmentations import *
from utils import custom_f1




def prepare_dataloader(df, fold, train_batch, valid_batch,img_sz,num_workers, data_root):

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

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler, scaler):
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
    
    print('train loss : {}, train accuracy : {}'.\
          format(np.round(avg_loss,6), np.round(accuracy*100,2)))

def valid_one_epoch(epoch, model, loss_fn, handwritten_val_loader, printed_val_loader, device, scheduler):
    model.eval()

    handwritten_lst_val_out = []
    handwritten_lst_val_label = []
    printed_lst_val_out = []
    printed_lst_val_label = []

    lst_val_out = []
    lst_val_label = []
    avg_val_loss = 0

    handwritten_status = tqdm(
        enumerate(handwritten_val_loader), 
        total=len(handwritten_val_loader), 
        desc = "Handwritten validation epoch " + str(epoch+1),
        position=0, leave=True)
    printed_status = tqdm(
        enumerate(printed_val_loader), 
        total=len(printed_val_loader), 
        desc = "Printed validation epoch " + str(epoch+1),
        position=0, leave=True)
    
    for step, (images, labels) in handwritten_status:
        val_images = images.to(device).float()
        val_labels = labels.to(device).long()

        val_preds = model(val_images)

        handwritten_lst_val_out += val_preds.argmax(1)
        handwritten_lst_val_label += val_labels

        lst_val_out += val_preds.argmax(1)
        lst_val_label += val_labels
        loss = loss_fn(val_preds, val_labels)
                       
        avg_val_loss += loss.item() / len(handwritten_status)

    for step, (images, labels) in printed_status:
        val_images = images.to(device).float()
        val_labels = labels.to(device).long()

        val_preds = model(val_images)

        printed_lst_val_out += val_preds.argmax(1)
        printed_lst_val_label += val_labels

        lst_val_out += val_preds.argmax(1)
        lst_val_label += val_labels
        loss = loss_fn(val_preds, val_labels)
                       
        avg_val_loss += loss.item() / len(printed_status)

    accuracy = accuracy_score(y_pred=torch.tensor(lst_val_out), y_true=torch.tensor(lst_val_label))

    handwritten_f1 = f1_score(y_pred=torch.tensor(handwritten_lst_val_out), y_true=torch.tensor(handwritten_lst_val_label), average="micro")
    printed_f1 = f1_score(y_pred=torch.tensor(printed_lst_val_out), y_true=torch.tensor(printed_lst_val_label), average="micro")
    mixed_f1 = f1_score(y_pred=torch.tensor(lst_val_out), y_true=torch.tensor(lst_val_label), average="micro")
    
    print("handwritten f1: {}".format(handwritten_f1))
    print("printed f1: {}".format(printed_f1))
    print("mixed f1: {}".format(mixed_f1))

    print('valid loss : {}, valid accuracy : {}'.\
          format(np.round(avg_val_loss, 6), np.round(accuracy*100,2)))

    return avg_val_loss


