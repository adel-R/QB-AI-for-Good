import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_metadata(path_to_csv_file, set = None):
    """
    This function loads metadata from a CSV file and optionally subsets the data based on a specified set.
    It also transforms the 'plume' column into binary values.

    Parameters:
    path_to_csv_file (str): The path to the CSV file containing the metadata.
    set (str, optional): The name of the set to subset the metadata. If None, all metadata are loaded without any subsetting. Defaults to None.

    Returns:
    DataFrame: A pandas DataFrame containing the metadata from the CSV file.
    """
    # load all metadata
    metadata = pd.read_csv(path_to_csv_file)

    # append .tif to all image paths
    metadata.path = metadata.path + ".tif"

    # subset to relevant data only
    if set is not None:
        metadata = metadata[metadata.set == set]
    
    # transform target to binary values
    metadata = metadata.assign(plume = lambda df_: [1 if y == "yes" else 0 for y in df_.plume])

    return metadata.reset_index(drop = True)


def trainval_split(metadata, val_size = 0.2):
    """
    This function splits a metadata DataFrame into two sets for training and validation, based on a specified split size.

    Parameters:
    metadata (DataFrame): A pandas DataFrame containing the metadata to be split.
    test_size (float, optional): The proportion (between 0 and 1) of the metadata to include in the validation set.

    Returns:
    tuple: A tuple containing two DataFrames. The first DataFrame is for training and the second DataFrame is for validation.
    """
    trn_metadata, val_metadata = train_test_split(metadata, test_size = val_size, random_state = 42) # fix seed for reproducible train test split
    return trn_metadata.reset_index(), val_metadata.reset_index()


def stratified_split(df, k, test_size = None):
    """
    Splits a dataframe into train, validation and optional test sets in a stratified manner.

    Parameters:
    df (pd.DataFrame): The dataframe to split.
    k (int): The number of folds to split the data into for cross-validation.
    test_size (float, optional): If provided, the proportion of data to include in the test split.

    Returns:
    folds (dict): A dictionary of train and validation dataframes for each fold.
    test_df (pd.DataFrame): A dataframe for the test set, only if test_size is provided.
    """

    # Obtain unique location identifiers
    unique_id_coords = df['id_coord'].unique()
    np.random.shuffle(unique_id_coords)  # shuffle the unique IDs

    # If a test_size is provided, create a separate test set
    if test_size is not None:
        test_id_coords = []
        test_count = 0
        test_size_total = int(test_size * len(df))

        # Loop over each unique ID, add to test set until test set size threshold is reached
        for id_coord in unique_id_coords:
            id_coord_df = df[df['id_coord'] == id_coord]
            if (test_count + len(id_coord_df) <= test_size_total):
                test_id_coords.append(id_coord)
                test_count += len(id_coord_df)
            else:
                break

        # Create test_df and the remaining df for train-val splitting
        test_df = df[df['id_coord'].isin(test_id_coords)]
        df_trainval = df[~df['id_coord'].isin(test_id_coords)]
        unique_id_coords = [id_coord for id_coord in unique_id_coords if id_coord not in test_id_coords]
    else:
        df_trainval = df

    folds = {}
    size_per_fold = len(df_trainval) / k

    # Create k folds
    for i in range(k):
        train_id_coords = []
        val_id_coords = []

        val_count = 0

        # Loop over each unique ID, add to validation set until validation set size threshold is reached
        for id_coord in unique_id_coords:
            id_coord_df = df_trainval[df_trainval['id_coord'] == id_coord]
            if (val_count + len(id_coord_df) <= size_per_fold) or (i == k - 1 and val_count < size_per_fold):
                val_id_coords.append(id_coord)
                val_count += len(id_coord_df)
            else:
                train_id_coords.append(id_coord)

        # Create train_df and val_df for the current fold
        train_df = df_trainval[df_trainval['id_coord'].isin(train_id_coords)]
        val_df = df_trainval[df_trainval['id_coord'].isin(val_id_coords)]

        folds[i] = {'train': train_df, 'val': val_df}

        # Update df_trainval and unique_id_coords for the next iteration
        df_trainval = df_trainval[~df_trainval['id_coord'].isin(val_id_coords)]
        unique_id_coords = [id_coord for id_coord in unique_id_coords if id_coord not in val_id_coords]

    # If test_size is provided, return folds and test_df; otherwise, return only folds
    if test_size is not None:
        return folds, test_df
    else:
        return folds


class CustomDataset(Dataset):
    """
    Dataset with input as path to metadata.csv file (so we don't have to load the whole Data in memory)
    """
    def __init__(self, metadata, transform=None, resize = False, apply_CLAHE = False, dir = None):
        # store metadata of dataset
        self.metadata = metadata

        # save transform
        self.transform = transform
        self.resize = resize
        self.apply_CLAHE = apply_CLAHE

        # store path to working directory
        if dir is not None:
            self.dir = dir + "/"
        else:
            self.dir = None
            
    def __getitem__(self, index):
        # fetch image
        img_path = self.metadata.loc[index].path
        if self.dir is not None:
            img_path = self.dir + img_path # needed to allow raytune to open images
        img = np.array(Image.open(img_path)) # shape (64, 64)

        # resize if needed
        if self.resize:
            img = cv2.resize(img, (224,224), interpolation = cv2.INTER_CUBIC)

        # apply CLAHE if needed
        if self.apply_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
            img = clahe.apply(img)

        # fetch ground truth label
        lbl = self.metadata.loc[index].plume

        # apply transform if available
        if self.transform:
            img = self.transform(image = img)["image"]
        
        return index, img, lbl

    def __len__(self):
        return len(self.metadata)

    def compute_mean_std(self, indices = None):
        # define iterator
        if indices is not None:
            iterator = np.array(list(self.metadata.path.values()))[indices]
        else:
            iterator = range(0, self.__len__())
        
        # initialise values for computation
        pixel_sum, pixel_sqsum = 0, 0
        count = 0

        # iterate over data
        for id in iterator:

            # fetch image
            img_path = self.metadata.loc[id].path
            if self.dir is not None:
                img_path = self.dir + img_path # needed to allow raytune to open images
            img = np.asarray(Image.open(img_path)).astype(np.float32)/255
            
            pixel_sum += np.sum(img)
            pixel_sqsum += np.sum(img ** 2)

            count += img.size # count how many pixels we have added to each channel within this step

        # compute mean
        pixel_mean = pixel_sum / count

        # compute std = sqrt(E[X^2] - (E[X])^2)
        pixel_std = (pixel_sqsum / count - pixel_mean ** 2) ** 0.5

        return pixel_mean, pixel_std