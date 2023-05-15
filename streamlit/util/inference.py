import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import cv2
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
    model.load_state_dict(torch.load(path))
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
