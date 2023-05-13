# import own scripts
try:
    import util.load_data as loadData
    import util.preprocess_data as prepData
except ModuleNotFoundError:
    import load_data as loadData
    import preprocess_data as prepData

# import external packages
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision


def get_datasets():

    # get transforms
    transform = prepData.get_transform() # to be applied during training

    # fetch metadata
    trnval_metadata = loadData.load_metadata("images/metadata.csv", set = "train")
    tst_metadata   = loadData.load_metadata("images/metadata.csv", set = "test")

    # perform train validation split
    trn_metadata, val_metadata = loadData.trainval_split(trnval_metadata, val_size = 0.2)

    # get pytorch datasets for modeling
    trn_dataset = loadData.CustomDataset(trn_metadata, transform = transform, apply_CLAHE = True)
    val_dataset = loadData.CustomDataset(val_metadata, transform = transform, apply_CLAHE = True)
    tst_dataset = loadData.CustomDataset(tst_metadata, transform = transform, apply_CLAHE = True)

    return trn_dataset, val_dataset, tst_dataset


def get_dataloader(dataset, batch_size, shuffle = True):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)


def get_device(model):
    # where we want to run the model (so this code can run on cpu, gpu, multiple gpus depending on system)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    return device, model.to(device)


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False


def init_training(config, use_model = None):
    # how many epochs we want to train for (at maximum)
    max_epochs = int(config["max_epochs"])
    
    # model initialisation
    weights = None
    param = None
    num_classes = 10
    
    if use_model is None:
        if config["pretrained"]:
            weights = "DEFAULT"
            num_classes = 1000 # pre-trained weights come from ImageNet (replace last layers with own classifier)

        if config["model"] == "ResNet18":
            model = torchvision.models.resnet18(weights = weights, progress = False, num_classes = num_classes)
            if weights:
                freeze_weights(model)
                model.fc = nn.Linear(512, 10) # replace last layer with own classifier
        elif config["model"] == "ResNet34":
            model = torchvision.models.resnet34(weights = weights, progress = False, num_classes = num_classes)
            if weights:
                freeze_weights(model)
                model.fc = nn.Linear(512, 10) # replace last layer with own classifier
        elif config["model"] == "efficientnet_V2_s":
            model = torchvision.models.efficientnet_v2_s(weights = weights, progress = False, num_classes = num_classes)
            if weights:
                freeze_weights(model)
                model.classifier[1] = nn.Linear(1280, 10) # replace last layer with own classifier

        # change 1st conv layer from 3 channel to 1 channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    else:
        model = use_model
            
    # initialise device
    device, model = get_device(model)

    # define optimizer
    param = model.parameters()