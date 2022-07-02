import os
import cv2
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Pytorch Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


ROOT_DIR = "../landmark_retrieval_dataset/"
TRAIN_DIR = ROOT_DIR + "train"
TEST_DIR = ROOT_DIR + "test"
MODEL_FILE = "ACC0.6051_epoch3.bin"
NUM_GPUS = 8


CONFIG = dict(
    seed=42,
    model_name='tf_mobilenetv3_small_100',
    train_batch_size=384,
    valid_batch_size=768,
    img_size=224,
    epochs=3,
    learning_rate=5e-4,
    scheduler=None,
    # min_lr = 1e-6,
    # T_max = 20,
    # T_0 = 25,
    # warmup_epochs = 0,
    weight_decay=1e-6,
    n_accumulate=1,
    # n_fold=10,
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


predict_transform = A.Compose(
    [A.Resize(CONFIG['img_size'], CONFIG['img_size']),
     A.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
         max_pixel_value=255.0,
         p=1.0
     ),
     ToTensorV2()], p=1.)


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


def prepare_loaders(df_train, df_valid, num_workers=0):
    train_dataset = LandmarkDataset(TRAIN_DIR, df_train, transforms=predict_transform)
    valid_dataset = LandmarkDataset(TRAIN_DIR, df_valid, transforms=predict_transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], num_workers=num_workers, pin_memory=True)
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


@torch.no_grad()
def predict_dataset(model, dataloader, device):
    model.eval()
    TARGETS = []
    PREDS = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        PREDS.append(joblib.load('label_encoder.pkl').classes_[preds.view(-1).cpu().detach().numpy()])
        TARGETS.append(labels.view(-1).cpu().detach().numpy())
    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    pred_acc = accuracy_score(TARGETS, PREDS)
    pred_f1_score = f1_score(TARGETS, PREDS, average='macro')
    return pred_acc, pred_f1_score


def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = predict_transform(image=img)["image"]
    img = img.to(CONFIG["device"], dtype=torch.float).unsqueeze_(0)
    output = model(img)
    index = output.data.cpu().numpy().argmax()
    return int(joblib.load('label_encoder.pkl').classes_[index])


set_seed(CONFIG['seed'])
model = LandmarkRetrievalModel(CONFIG['model_name'])
model.to(CONFIG['device'])
model.load_state_dict(torch.load(MODEL_FILE))

df_landmarks = pd.read_csv(f"{ROOT_DIR}train_label_to_landmark.csv")
df = pd.read_csv(f"{ROOT_DIR}/train.csv")
df['file_path'] = df['id'].apply(lambda id: f"{TRAIN_DIR}/{id[0]}/{id[1]}/{id[2]}/{id}.jpg")
df_train, df_valid = train_test_split(df, test_size=0.1, stratify=df.landmark_id, shuffle=True, random_state=CONFIG['seed'])
train_loader, valid_loader = prepare_loaders(df_train, df_valid, num_workers=NUM_GPUS)

train_acc, train_f1_score = predict_dataset(model, train_loader, CONFIG['device'])
print(f"Training set accuracy: {train_acc}")
print(f"Training set F1 Score: {train_f1_score}")

val_acc, val_f1_score = predict_dataset(model, valid_loader, CONFIG['device'])
print(f"Validation set accuracy: {val_acc}")
print(f"Validation set F1 Score: {val_f1_score}")
