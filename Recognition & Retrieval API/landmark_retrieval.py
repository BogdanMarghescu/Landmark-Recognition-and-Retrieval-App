import os
import torch
import torch.nn as nn
import timm
import random
import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA, IncrementalPCA
from datetime import datetime

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
    dev_size=0.1,
    num_classes=81313,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition='GOOGL',
    _wandb_kernel='deb'
)

MODEL_NAME = "ACC0.6051_epoch3.bin"
ROOT_DIR = "landmark_retrieval_dataset/"
TRAIN_DIR = ROOT_DIR + "train"
TEST_DIR = ROOT_DIR + "test"
MODEL_FILE = "ACC0.6051_epoch3.bin"
NUM_GPUS = 8


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


predict_transform = A.Compose(
    [A.Resize(CONFIG['img_size'], CONFIG['img_size']),
     A.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
         max_pixel_value=255.0,
         p=1.0
     ),
     ToTensorV2()], p=1.)


def get_image_embeddings(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = predict_transform(image=img)["image"]
    img = img.to(CONFIG["device"], dtype=torch.float).unsqueeze_(0)
    output = model(img)
    softmax = nn.Softmax(dim=1)
    output = softmax(output).data.cpu().numpy().flatten()
    return output


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()
model = LandmarkRetrievalModel(CONFIG['model_name'])
model.to(CONFIG['device'])
model.load_state_dict(torch.load(MODEL_NAME))
model.eval()
df_landmarks = pd.read_csv(f"{ROOT_DIR}train_label_to_landmark.csv")
df = pd.read_csv(f"{ROOT_DIR}train.csv")
df = df.merge(df_landmarks, on='landmark_id')
df['file_path'] = df['id'].apply(lambda id: f"{TRAIN_DIR}/{id[0]}/{id[1]}/{id[2]}/{id}.jpg")

# Train the PCA model for dimensionality reduction
embeddings_list = []
length = 20000
bar = tqdm(df.sample(length).itertuples(index=False), total=length)
for images in bar:
    embeddings_list.append(get_image_embeddings(images.file_path, model))
embeddings_list = np.array(embeddings_list)
start_time_pca = datetime.now()
pca = IncrementalPCA(n_components=100)
pca.fit(embeddings_list)
end_time_pca = datetime.now()
print(f"PCA training time: {end_time_pca - start_time_pca}")
pickle.dump(pca, open("pca.pkl", "wb"))

# Train the K-Means Algorithm
pca = pickle.load(open("pca.pkl", "rb"))
embeddings_list = []
bar = tqdm(df.itertuples(index=False), total=df.shape[0])
for images in bar:
    embeddings = np.expand_dims(get_image_embeddings(images.file_path, model), 0)
    embeddings_list.append(pca.transform(embeddings).flatten())
embeddings_list = np.array(embeddings_list)
print(embeddings_list.shape)
start_time_kmeans = datetime.now()
print(f"K-Means training start time: {start_time_kmeans}")
kmeans = MiniBatchKMeans(n_clusters=CONFIG["num_classes"], verbose=1)
kmeans.fit(embeddings_list)
end_time_kmeans = datetime.now()
print(f"K-Means training end time: {end_time_kmeans}")
print(f"K-Means training time: {end_time_kmeans - start_time_kmeans}")
pickle.dump(kmeans, open("kmeans.pkl", "wb"))
