import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transform(visualize = False):
    """
    This function creates and returns a data augmentation and normalization pipeline using the Albumentations library.
    If 'visualize' is set to True, the normalization and tensor conversion steps are not included in the returned pipeline
    (otherwise image cannot be interpreted by humans anymore).

    Parameters:
    visualize (bool, optional): A flag indicating whether the transformation pipeline is used for visualization.

    Returns:
    A.Compose: An Albumentations composition of transformations.
    """
    # get preprocessing settings from .ini
    MEAN, STD = [60.0644191863925 / 65535.0], [58.84460681054689 / 65535.0]

    trn_transform_list = [
            # necessary as we work with uint16 images (https://albumentations.ai/docs/examples/example_16_bit_tiff/)
            A.ToFloat(max_value = 65535.0),
            # random permutations
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            # sharpen image
            A.UnsharpMask(p = 0.5),
            # contrast, brightness
            A.RandomBrightnessContrast(p = 0.5),
            # geometric transformations
            A.OneOf(
                [A.Affine(scale = (0.9, 1.1), translate_percent = (0.9, 1.1), interpolation = cv2.INTER_CUBIC),
                A.ElasticTransform(interpolation = cv2.INTER_CUBIC, p = 0.5)], p = 0.5
            ),
            # normalize & convert data to tensor
            A.Normalize(mean = MEAN, std = STD),
            ToTensorV2()
    ]

    valtst_transform_list = [
            # necessary as we work with uint16 images (https://albumentations.ai/docs/examples/example_16_bit_tiff/)
            A.ToFloat(max_value = 65535.0),
            # color augmentations
            A.UnsharpMask(p = 1), # always sharpen the image
            # normalize & convert data to tensor
            A.Normalize(mean = MEAN, std = STD),
            ToTensorV2()
    ]
    
    if visualize:
        trn_transform_list = trn_transform_list[:-2]  # Remove the last two transforms
        trn_transform_list.append(A.FromFloat(max_value=65535.0)) 
        valtst_transform_list = valtst_transform_list[:-2]
        valtst_transform_list.append(A.FromFloat(max_value=65535.0))
        return A.Compose(trn_transform_list), A.Compose(valtst_transform_list)
    
    return A.Compose(trn_transform_list), A.Compose(valtst_transform_list)