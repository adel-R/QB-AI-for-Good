# import external packages
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualize(image, lbl, transform = False, pred = None, likelihood = None):
    """
    This function visualizes an image, optionally applies a transformation to the image,
    and annotates the image based on the provided label, prediction, and likelihood information.

    Parameters:
    image (Image): An instance of PIL Image class. This is the image to be visualized.
    lbl (int): A binary value (0 or 1) indicating the ground truth label of the image. 1 indicates presence of plume, 0 indicates absence.
    transform (int): A binary value (0 or 1) indicating whether or not the image was transformed.
    pred (int, optional): A binary value (0 or 1) indicating the predicted label of the image.
    likelihood (float, optional): A float value between 0 and 1 representing the likelihood (confidence) of the prediction.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 6))
    fig.tight_layout(w_pad = 5)

    # transform image (if necessary) & define titles
    transform_description = " (raw)"

    if transform:
        transform_description = " (transformed)"

    # create ground truth string that we will add to title
    lbl_description = "containing plume" if lbl == 1 else "containing no plume"

    # create prediction strings that we will add to title (if available)
    pred_description = ", predicted plume" if pred == 1 else 0 if pred == 0 else ""
    likelihood_description = f" with score of {round(likelihood, 3) * 100} % " if (pred is not None) and likelihood is not None else ""

    # create title
    title = f"Image{transform_description}, {lbl_description}{pred_description}{likelihood_description}"

    ax.set_title(title)

    # plot images
    print(image.size)
    ax.imshow(image, cmap = "gray_r")