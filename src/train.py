import parser
from sklearn.preprocessing import LabelEncoder

from engine import prepare_dataloader, train_one_epoch, valid_one_epoch
from utils import EarlyStopping
from models.ViT import ViT


from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable


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
    df = pd.read_csv(TRAIN_CSV)

    labelencoder= LabelEncoder() 
    df['class'] = labelencoder.fit_transform(df['class']) # Convert labels to class number

    train_loader, val_loader = prepare_dataloader(
        train_df, opt.fold, train_batch, valid_batch, opt.img_size, opt.num_workers, data_root=images_path)

    # --------------------------------------
    # Models
    # --------------------------------------
  
    model = ViT(model_name=opt.model_name, n_classes=n_classes, pretrained=True).to(device)

    if opt.weights is not None:
        cp = torch.load(opt.weights)
        model.load_state_dict(cp['model_state_dict'])
    
    # -------------------------------------------
    # Setup optimizer, scheduler, criterion loss
    # -------------------------------------------

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    early_stopping = EarlyStopping(patience=opt.patience)
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

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint["scheduler"])


    # --------------------------------------
    # Start training
    # --------------------------------------
    print("[INFO] Start training...")
    for epoch in range(start_epoch, epochs):
        train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)
        save_model(model, optimizer, scheduler, fold, epoch, save_every=False)
        with torch.no_grad():
            val_loss = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None)
            early_stopping(val_loss, model, optimizer, scheduler, fold, epoch)
            if early_stopping.early_stop:
                break
            
    del model, optimizer, train_loader, val_loader, scheduler, scaler
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/content/train_images', help='image directory')
    parser.add_argument('--train_csv', type=str,default='/content/Shopee/split/group5folds.csv', help='group5folds')
    parser.add_argument('--fold', type=int, help='fold number')
    parser.add_argument('--seed', type=int, default=2021, help='for reproduce')
    parser.add_argument('--epochs', type=int, default=20, help='number of epoch')
    parser.add_argument('--train_bs', type=int, default=32, help='train batch size')
    parser.add_argument('--valid_bs', type=int, default=64, help='validation batch size')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--work_dir', type=str, default='CLIP/fold0', help='path to save model')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--weights', type=str, help='add weight path')
    parser.add_argument('--model_name', type=str, default='tf_efficientnet_b4', help="ViT-B-32, RN50, RN50x4, RN101")
    parser.add_argument('--patience', type=int, default=5, help="set early stopping patience")
    parser.add_argument('--loss', type=str, default="triplet", help='arcface, cosface, adacos, triplet')
    parser.add_argument('--img_size', type=int, default=224, help='resize the image')
    opt = parser.parse_args()

    seed_torch(seed=opt.seed)
    run_training(opt)