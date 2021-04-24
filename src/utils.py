import cv2
import random
import torch
import os
import numpy as np

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def save_model(model, optimizer, scheduler, fold, epoch, save_every=False, best=False):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    if save_every == True:
        if not (os.path.isdir('./saved_model')): os.mkdir('./saved_model')
        torch.save(state, './saved_model/vit_fold_{}_epoch_{}'.format(fold, epoch+1))
    if best == True:
        if not (os.path.isdir('./best_model')): os.mkdir('./best_model')
        torch.save(state, './best_model/vit_fold_{}_epoch_{}'.format(fold, epoch+1))
def custom_f1(df, fold):

    return 

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimizer, scheduler, fold, epoch):
        if self.val_loss_min == np.Inf:
            self.val_loss_min = val_loss
        elif val_loss > self.val_loss_min:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                print('Early Stopping - Fold {} Training is Stopping'.format(fold))
                self.early_stop = True
        else:  # val_loss < val_loss_min
            save_model(model, optimizer, scheduler, fold, epoch, best=True)
            print('*** Validation loss decreased ({} --> {}).  Saving model... ***'.\
                  format(round(self.val_loss_min, 6), round(val_loss, 6)))
            self.val_loss_min = val_loss
            self.counter = 0
        