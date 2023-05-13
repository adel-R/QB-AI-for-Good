import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    MEANS, STDS = [1., 1., 1.], [1., 1., 1.]

    transform_list = [
            # necessary as we work with uint16 images (https://albumentations.ai/docs/examples/example_16_bit_tiff/)
            A.ToFloat(max_value = 65535.0),
            # random permutations
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            # color augmentations
            A.UnsharpMask(p = 1), # always sharpen the image
            # normalize & convert data to tensor
            A.Normalize(mean=MEANS, std=STDS),
            ToTensorV2()
    ]
    
    if visualize:
        transform_list = transform_list[:-2]  # Remove the last two transforms
        transform_list.append(A.FromFloat(max_value=65535.0)) 
        return A.Compose(transform_list)
    
    return A.Compose(transform_list)