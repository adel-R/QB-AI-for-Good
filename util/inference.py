import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import cv2

# import own scripts
import util.preprocess_data as prepData

def load_resnet34(path = "models/best_ResNet34.pt"):
    weights = "DEFAULT"

    # initialize model
    model = torchvision.models.resnet34(weights=weights)
    model.fc = nn.Linear(512, 1)  # replace last layer with own classifier

    # change 1st conv layer from 3 channel to 1 channel (ResNets were pretrained using 3channel RGB images)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # load weights
    model.load_state_dict(torch.load(path))
    return model


def prep_single_image(path_to_img):
    # Getting transformations
    _, valtst_transform = prepData.get_transform()

    # Opening image
    img = np.array(Image.open(path_to_img))  # shape (64, 64)

    # Resizing
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Applying CLACHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = valtst_transform(image=img)["image"]

    return img


def infer(model, path_to_img, device, multiple=False):
    """
    returns predictions for given model, if multiple False, expected
    #TODO multiple images inference
    """
    # Transforming image to tensor
    img_tensor = prep_single_image(path_to_img)

    # Inference
    model.eval()
    prediction = model(img_tensor.unsqueeze(0).to(device))
    prob = torch.sigmoid(prediction)

    # Converting probabilities into predictions
    if prob.item() > .5:
        output = 1
    else:
        output = 0

    return prob.item(), output





