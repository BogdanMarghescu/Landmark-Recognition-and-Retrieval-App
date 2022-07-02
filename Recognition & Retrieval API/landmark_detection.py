import joblib
import torch
import torch.nn as nn
import timm
from flask import Flask, request, Response
from requests.sessions import Session
import jsonpickle
import numpy as np
import cv2
import pandas as pd
import googlemaps
import wikipedia
import json
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask("Landmark Detection App")

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
    n_fold=10,
    num_classes=81313,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition='GOOGL',
    _wandb_kernel='deb'
)
MODEL_NAME = "ACC0.6051_epoch3.bin"


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


def get_image_embeddings(img, model):
    img = predict_transform(image=img)["image"]
    img = img.to(CONFIG["device"], dtype=torch.float).unsqueeze_(0)
    output = model(img)
    softmax = nn.Softmax(dim=1)
    output = softmax(output).data.cpu().numpy().flatten()
    return output


def predict_image(embeddings):
    return int(joblib.load('label_encoder.pkl').classes_[embeddings.argmax()])


def retrieve_landmarks(embeddings, df_images):
    pca = pickle.load(open("pca.pkl", "rb"))
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    prediction = kmeans.predict(pca.transform(np.expand_dims(embeddings, 0)))[0]
    return " ".join(df_images.iloc[np.where(kmeans.labels_ == prediction)]['title'].apply(get_image_url).values.tolist())


def wiki_url_with_suggestion(landmark, auto_suggest=True):
    url = ''
    try:
        url = wikipedia.page(landmark, auto_suggest=auto_suggest).url
    except:
        url = ''
    finally:
        return url


def wiki_url(landmark):
    url = wiki_url_with_suggestion(landmark, auto_suggest=False)
    if url == '':
        url = wiki_url_with_suggestion(landmark)
    return url


def get_image_url(wikimedia_title):
    PARAMS = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
        "titles": wikimedia_title
    }
    DATA = Session().get(url="https://commons.wikimedia.org/w/api.php", params=PARAMS).json()["query"]["pages"]
    if "imageinfo" in DATA[list(DATA.keys())[0]]:
        return DATA[list(DATA.keys())[0]]["imageinfo"][0]["url"]
    else:
        return ""


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['picture'].stream.read()
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    embeddings = get_image_embeddings(img, model)
    label = predict_image(embeddings)
    landmark_name = df_landmarks.loc[label]['name']
    similar_images_list = retrieve_landmarks(embeddings, df_images)
    geocode_result = gmaps.geocode(landmark_name)[0]
    geocode_result['image_class_id'] = label
    geocode_result['landmark_name'] = landmark_name
    geocode_result['wiki_url'] = wiki_url(landmark_name)
    geocode_result["similar_images"] = similar_images_list
    print(f"\nPredicted label: {label}\nPredicted landmark name: {landmark_name}")
    print(json.dumps(geocode_result, indent=4, ensure_ascii=False))
    response_pickled = jsonpickle.encode(geocode_result)
    return Response(response=response_pickled, status=200, mimetype="application/json")


model = LandmarkRetrievalModel(CONFIG['model_name'])
model.to(CONFIG['device'])
model.load_state_dict(torch.load(MODEL_NAME))
print(f"Landmark Retrieval Model Structure:\n{str(model.eval())}\n")
df_landmarks = pd.read_csv("train_label_to_landmark.csv")
df_landmark_attribution = pd.read_csv("train_attribution.csv")
df_images = pd.read_csv("train.csv")
df_images = df_images.merge(df_landmarks, on='landmark_id')
df_images = df_images.merge(df_landmark_attribution, on='id')
api_key = open('maps_api_key.txt').readline()
gmaps = googlemaps.Client(key=api_key)
app.run(host="0.0.0.0", port=5000)
