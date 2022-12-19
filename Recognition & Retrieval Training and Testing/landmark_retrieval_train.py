import os
import cv2
import time
import wandb
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Training Configuration
ROOT_DIR = "landmark_retrieval_dataset/"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
RUN_NAME = "train12"
CONFIG = dict(
    num_gpus=8,
    seed=42,
    model_name='tf_mobilenetv3_small_100',
    train_batch_size=384,
    valid_batch_size=768,
    img_size=224,
    epochs=5,
    learning_rate=5e-4,
    weight_decay=1e-6,
    n_accumulate=1,
    # n_fold=10,
    valid_size=0.1,
    num_classes=81313,
    competition='Google Landmark Retrieval',
    _wandb_kernel='deb',
)


def set_seed():
    pl.seed_everything(CONFIG['seed'], workers=True)
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed(CONFIG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(CONFIG['seed'])


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


def prepare_loaders(df_train, df_valid):
    train_dataset = LandmarkDataset(TRAIN_DIR, df_train, transforms=data_transforms['train'])
    valid_dataset = LandmarkDataset(TRAIN_DIR, df_valid, transforms=data_transforms['valid'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], num_workers=4*CONFIG['num_gpus'], shuffle=True, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], num_workers=4*CONFIG['num_gpus'], shuffle=False, pin_memory=True, persistent_workers=True)
    return train_loader, valid_loader


class LandmarkRetrievalModel(pl.LightningModule):
    def __init__(self):
        super(LandmarkRetrievalModel, self).__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.model = timm.create_model(CONFIG['model_name'], pretrained=True)
        self.n_features = self.model.classifier.in_features
        self.model.reset_classifier(0)
        self.fc = nn.Linear(self.n_features, CONFIG['num_classes'])

    def forward(self, x):
        features = self.model(x)
        output = self.fc(features)
        return output
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        train_loss = self.criterion(y_hat, y)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        preds = torch.argmax(y_hat, dim=1)
        train_acc = accuracy(preds, y, 'multiclass', num_classes=CONFIG['num_classes'])
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': train_loss, 'train_acc': train_acc}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        valid_loss = self.criterion(y_hat, y)
        self.log('valid_loss', valid_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        preds = torch.argmax(y_hat, dim=1)
        valid_acc = accuracy(preds, y, 'multiclass', num_classes=CONFIG['num_classes'])
        self.log('valid_acc', valid_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': valid_loss, 'valid_acc': valid_acc}


def run_training(model, train_loader, valid_loader, wandb_logger, fold=None):
    start = time.time()
    savedir = os.path.join('Models', RUN_NAME)
    if fold is not None:
        savedir = os.path.join(savedir, f'fold{fold}')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    checkpoint_callback = ModelCheckpoint(monitor='valid_acc', mode='max', save_last=True, verbose=True, dirpath=savedir)
    trainer = pl.Trainer(
        strategy = DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        devices=CONFIG['num_gpus'],
        max_epochs=CONFIG['epochs'],
        callbacks=[checkpoint_callback],
        precision=16,
        amp_backend='native',
        num_sanity_val_steps=0,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, valid_loader)
    end = time.time()
    time_elapsed = end - start
    print(f"Training complete in {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {(time_elapsed % 3600) % 60:.0f}s")


def main():
    # WandB Initialization
    api_key = open('wandb_api_key.txt').readline()
    try:
        wandb.login(key=api_key)
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    # Read the data
    set_seed()
    df = pd.read_csv(os.path.join(ROOT_DIR, 'train.csv'))
    le = LabelEncoder()
    df.landmark_id = le.fit_transform(df.landmark_id)
    joblib.dump(le, 'label_encoder.pkl')
    df['file_path'] = df['id'].apply(get_train_file_path)

    # Train the model
    if "n_fold" in CONFIG:
        # Stratified KFold cross-validation training
        skf = StratifiedKFold(n_splits=CONFIG["n_fold"], shuffle=True, random_state=CONFIG['seed'])
        for fold, (train_ids, test_ids) in enumerate(skf.split(df, df.landmark_id)):
            wandb_logger = WandbLogger(project='Landmark Retrieval', config=CONFIG, job_type='Train', anonymous=anony, group=RUN_NAME, name=f"fold{fold}_{RUN_NAME}", log_model=True)
            train_loader, valid_loader = prepare_loaders(df.iloc[train_ids], df.iloc[test_ids])
            model = LandmarkRetrievalModel()
            print(f"\n\nFold {fold}:")
            run_training(model, train_loader, valid_loader, wandb_logger, fold)
            wandb.finish()
    else:
        # Simple cross-validation training
        wandb_logger = WandbLogger(project='Landmark Retrieval', config=CONFIG, job_type='Train', anonymous=anony, group=RUN_NAME, name=RUN_NAME, log_model=True)
        df_train, df_valid = train_test_split(df, test_size=CONFIG['valid_size'], stratify=df.landmark_id, shuffle=True, random_state=CONFIG['seed'])
        train_loader, valid_loader = prepare_loaders(df_train, df_valid)
        model = LandmarkRetrievalModel()
        run_training(model, train_loader, valid_loader, wandb_logger)
        wandb.finish()


if __name__ == '__main__':
    main()
