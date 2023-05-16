import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import cv2
import os
# import own scripts
import util.preprocess_data as prepData


def load_resnet34(path="models/best_ResNet34.pt"):
    weights = "DEFAULT"

    # initialize model
    model = torchvision.models.resnet34(weights=weights)
    model.fc = nn.Linear(512, 1)  # replace last layer with own classifier

    # change 1st conv layer from 3 channel to 1 channel (ResNets were pretrained using 3channel RGB images)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # load weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def load_mobilenet_v3_large(path):
    # Creating model
    model = torchvision.models.mobilenet_v3_large(progress=False, num_classes=1)
    model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                     bias=False)  # change 1st conv layer from 3 channel to 1 channel
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    return model


def prep_single_image(path_to_img, apply_CLACHE = False):
    # Getting transformations
    _, valtst_transform = prepData.get_transform()

    # Opening image
    img_raw = np.array(Image.open(path_to_img))  # shape (64, 64)

    # Resizing
    img_raw = cv2.resize(img_raw, (224, 224), interpolation=cv2.INTER_CUBIC)
    # copy of image
    img = img_raw

    # Applying CLACHE
    if apply_CLACHE:
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        img = clahe.apply(img)

    img = valtst_transform(image=img)["image"]

    return img, img_raw


def infer(model, path_to_img, device, multiple=False):
    """
    returns predictions for given model, if multiple False, expected
    #TODO multiple images inference
    """
    # Transforming image to tensor
    img_tensor, img_raw = prep_single_image(path_to_img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    model.eval()
    prediction = model(img_tensor)
    prob = torch.sigmoid(prediction)

    # Converting probabilities into predictions
    if prob.item() > .5:
        output = 1
    else:
        output = 0

    return prob.item(), output


def infer_mobilenet(path_to_models, path_to_img, device):
    # Get the list of files in the directory
    file_list = os.listdir(path_to_models)

    # Transforming image to tensor
    img_tensor, img_raw = prep_single_image(path_to_img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    probs = []

    # Infer
    for model_path in file_list:
        if model_path.split(".")[-1] == "pt":
            model = load_mobilenet_v3_large(path_to_models + model_path)
        else:
            continue

        # Model to correct device
        model.to(device)

        # Predict
        model.eval()
        logit = model(img_tensor)
        prob = torch.sigmoid(logit)
        probs.append(prob.item())

    # Final prediction
    prob = np.mean(probs)
    best_model = np.argmax(probs)

    # Converting probabilities into predictions
    if prob > .5:
        output = 1
    else:
        output = 0

    return prob, output, best_model