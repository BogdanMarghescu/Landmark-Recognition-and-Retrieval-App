import os
import gc
import cv2
import copy
import time
import random
import wandb

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import timm

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Training Configuration
ROOT_DIR = "../landmark_retrieval_dataset/"
TRAIN_DIR = ROOT_DIR + "train"
TEST_DIR = ROOT_DIR + "test"
RUN_NAME = "train05"
NUM_GPUS = 8
CONFIG = dict(
    seed=42,
    model_name='tf_mobilenetv3_small_100',
    train_batch_size=384,
    valid_batch_size=768,
    img_size=224,
    epochs=5,
    learning_rate=5e-4,
    scheduler=None,
    # min_lr = 1e-6,
    # T_max = 20,
    # T_0 = 25,
    # warmup_epochs = 0,
    weight_decay=1e-6,
    n_accumulate=1,
    # n_fold=10,
    dev_size=0.2,
    num_classes=81313,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition='GOOGL',
    _wandb_kernel='deb'
)


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run. This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_train_file_path(id):
    return f"{TRAIN_DIR}/{id[0]}/{id[1]}/{id[2]}/{id}.jpg"


class LandmarkDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None):
        self.root_dir = root_dir
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['landmark_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img, label


data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.),

    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
}


def prepare_loaders(df_train, df_valid, num_workers=0):
    train_dataset = LandmarkDataset(TRAIN_DIR, df_train, transforms=data_transforms['train'])
    valid_dataset = LandmarkDataset(TRAIN_DIR, df_valid, transforms=data_transforms['valid'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], num_workers=num_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, valid_loader


class LandmarkRetrievalModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(LandmarkRetrievalModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.reset_classifier(0)
        self.fc = nn.Linear(self.n_features, CONFIG['num_classes'])

    def forward(self, x):
        features = self.model(x)  # features = (bs, embedding_size)
        output = self.fc(features)  # outputs = (bs, num_classes)
        return output

    def extract_features(self, x):
        features = self.model(x)  # features = (bs, embedding_size)
        return features


def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], T_mult=1, eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] is None:
        return None
    return scheduler


def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / CONFIG['n_accumulate']

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            for p in model.parameters():
                p.grad = None

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    TARGETS = []
    PREDS = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        PREDS.append(preds.view(-1).cpu().detach().numpy())
        TARGETS.append(labels.view(-1).cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    val_acc = accuracy_score(TARGETS, PREDS)
    val_f1_score = f1_score(TARGETS, PREDS, average='macro')
    gc.collect()

    return epoch_loss, val_acc, val_f1_score


def run_training(model, optimizer, scheduler, train_loader, valid_loader, fold=None):
    wandb.watch(model, log_freq=100)    # To automatically log gradients
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}\n")

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_acc = 0
    history = defaultdict(list)

    for epoch in range(1, CONFIG['epochs'] + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, train_loader, device=CONFIG['device'], epoch=epoch)
        val_epoch_loss, val_epoch_acc, val_f1_score = valid_one_epoch(model, optimizer, valid_loader, device=CONFIG['device'], epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid Acc'].append(val_epoch_acc)
        history['Valid F1 Score'].append(val_f1_score)

        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        wandb.log({"Valid Acc": val_epoch_acc})
        wandb.log({"Valid F1 Score": val_f1_score})

        print(f'Valid Acc: {val_epoch_acc}')
        print(f'Valid F1 Score: {val_f1_score}')

        # deep copy the model
        if val_epoch_acc >= best_epoch_acc:
            print(f"Validation Acc Improved ({best_epoch_acc} ---> {val_epoch_acc})")
            best_epoch_acc = val_epoch_acc
            run.summary["Best Accuracy"] = best_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if fold is None:
                PATH = f"ACC{best_epoch_acc:.4f}_epoch{epoch:.0f}.bin"
            else:
                PATH = f"ACC{best_epoch_acc:.4f}_epoch{epoch:.0f}_fold{fold}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved to {PATH}")

        print()

    end = time.time()
    time_elapsed = end - start
    print(f"Training complete in {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {(time_elapsed % 3600) % 60:.0f}s")
    print(f"Best ACC: {best_epoch_acc:.4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


# WandB Initialization
api_key = open('wandb_api_key.txt').readline()
try:
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

# Read the data
set_seed(CONFIG['seed'])
df = pd.read_csv(f"{ROOT_DIR}train.csv")
le = LabelEncoder()
df.landmark_id = le.fit_transform(df.landmark_id)
joblib.dump(le, 'label_encoder.pkl')
df['file_path'] = df['id'].apply(get_train_file_path)

if "n_fold" in CONFIG:
    # Stratified KFold cross-validation training
    skf = StratifiedKFold(n_splits=CONFIG["n_fold"], shuffle=True, random_state=CONFIG['seed'])
    for fold, (train_ids, test_ids) in enumerate(skf.split(df, df.landmark_id)):
        run = wandb.init(project='Landmark Retrieval', config=CONFIG, job_type='Train', anonymous='must', group=RUN_NAME, name=f"fold{fold}_{RUN_NAME}")
        train_loader, valid_loader = prepare_loaders(df.iloc[train_ids], df.iloc[test_ids], num_workers=NUM_GPUS)
        model = LandmarkRetrievalModel(CONFIG['model_name'])
        model.to(CONFIG['device'])
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        scheduler = fetch_scheduler(optimizer)
        print(f"\n\nFold {fold}:")
        model, history = run_training(model, optimizer, scheduler, train_loader, valid_loader, fold)
        run.finish()
else:
    # Simple cross-validation training
    run = wandb.init(project='Landmark Retrieval', config=CONFIG, job_type='Train', anonymous='must', group=RUN_NAME, name=RUN_NAME)
    df_train, df_valid = train_test_split(df, test_size=CONFIG['dev_size'], stratify=df.landmark_id, shuffle=True, random_state=CONFIG['seed'])
    train_loader, valid_loader = prepare_loaders(df_train, df_valid, num_workers=NUM_GPUS)
    model = LandmarkRetrievalModel(CONFIG['model_name'])
    model.to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer)
    model, history = run_training(model, optimizer, scheduler, train_loader, valid_loader)
    run.finish()
