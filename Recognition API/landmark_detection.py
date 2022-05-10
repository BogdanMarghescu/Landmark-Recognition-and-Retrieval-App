import joblib
import torch
import torch.nn as nn
import timm
from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import pandas as pd
import googlemaps
import json
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


def predict_image(img, model):
    img = predict_transform(image=img)["image"]
    img = img.to(CONFIG["device"], dtype=torch.float).unsqueeze_(0)
    output = model(img)
    index = output.data.cpu().numpy().argmax()
    return int(joblib.load('label_encoder.pkl').classes_[index])


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['picture'].stream.read()
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    label = predict_image(img, model)
    landmark_name = df_landmarks.loc[label]['name']
    geocode_result = gmaps.geocode(landmark_name)[0]
    geocode_result['image_class_id'] = label
    geocode_result['landmark_name'] = landmark_name
    print(f"\nPredicted label: {label}\nPredicted landmark name: {landmark_name}")
    print(json.dumps(geocode_result, indent=4, ensure_ascii=False))
    response_pickled = jsonpickle.encode(geocode_result)
    return Response(response=response_pickled, status=200, mimetype="application/json")


model = LandmarkRetrievalModel(CONFIG['model_name'])
model.to(CONFIG['device'])
model.load_state_dict(torch.load("ACC0.6051_epoch3.bin"))
print(f"Landmark Retrieval Model Structure:\n{str(model.eval())}\n")
df_landmarks = pd.read_csv("train_label_to_landmark.csv")
api_key = open('maps_api_key.txt').readline()
gmaps = googlemaps.Client(key=api_key)
app.run(host="0.0.0.0", port=5000)
