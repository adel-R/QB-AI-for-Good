# import own scripts
try:
    import util.load_data as loadData
    import util.preprocess_data as prepData
except ModuleNotFoundError:
    import load_data as loadData
    import preprocess_data as prepData

# import external packages
## data handling
import os
import numpy as np
## modeling
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
## evaluation
from sklearn.metrics import accuracy_score, roc_auc_score
## hyperparam optimization
from ray import air, tune
from ray.air import session
from ray.tune import JupyterNotebookReporter


def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # same for pytorch
    random_seed = 1 # or any of your favorite number 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def init_data(config):

    # get transforms
    trn_transform, valtst_transform = prepData.get_transform() # to be applied during training

    # fetch metadata
    trnval_metadata = loadData.load_metadata(config["dir"] + "\\images\\metadata.csv", set = "train")
    if config["official_test"]:
        tst_metadata    = loadData.load_metadata(config["dir"] + "\\images\\metadata.csv", set = "test")

    # perform train val test split
    if config["official_test"]:
        folds, _ = loadData.stratified_split(trnval_metadata, k = config["k_cv"])
    else:
        folds, tst_metadata = loadData.stratified_split(trnval_metadata, k = config["k_cv"], test_size = config["artificial_test_size"])

    # get pytorch datasets for modeling
    trn_datasets = []
    val_datasets = []
    for k in range(config["k_cv"]):
        trn_dataset = loadData.CustomDataset(folds[k]["train"].reset_index(), transform = trn_transform, resize = config["resize"], apply_CLAHE = config["apply_CLAHE"], dir = config["dir"])
        val_dataset = loadData.CustomDataset(folds[k]["val"].reset_index(),   transform = valtst_transform, resize = config["resize"], apply_CLAHE = config["apply_CLAHE"], dir = config["dir"])
        trn_datasets.append(trn_dataset)
        val_datasets.append(val_dataset)

    tst_dataset = loadData.CustomDataset(tst_metadata.reset_index(), transform = valtst_transform, resize = config["resize"], apply_CLAHE = config["apply_CLAHE"], dir = config["dir"])

    # get pytorch dataloaders
    trnloaders = []
    valloaders = []
    for trn_dataset, val_dataset in zip(trn_datasets, val_datasets):
        trnloader = get_dataloader(trn_dataset, config["batch_size"], shuffle = True)
        valloader = get_dataloader(val_dataset, config["batch_size"], shuffle = False)
        trnloaders.append(trnloader)
        valloaders.append(valloader)
    
    tstloader = get_dataloader(tst_dataset, config["batch_size"], shuffle = False)

    return trnloaders, valloaders, tstloader


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
    num_classes = 1
    
    if use_model is None:
        if config["pretrained"]:
            weights = "DEFAULT"
            num_classes = 1000 # pre-trained weights come from ImageNet (replace last layers with own classifier)

        if config["model"] == "ResNet18":
            model = torchvision.models.resnet18(weights = weights, progress = False, num_classes = num_classes)
            if weights:
                freeze_weights(model)
                model.fc = nn.Linear(512, 1) # replace last layer with own classifier
        elif config["model"] == "ResNet34":
            model = torchvision.models.resnet34(weights = weights, progress = False, num_classes = num_classes)
            if weights:
                freeze_weights(model)
                model.fc = nn.Linear(512, 1) # replace last layer with own classifier

        # change 1st conv layer from 3 channel to 1 channel (ResNets were pretrained using 3channel RGB images)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    else:
        model = use_model
    
    # initialise device
    device, model = get_device(model)

    # initialise optimizer
    optimizer = torch.optim.SGD(params = model.parameters(), lr = config["lr"], momentum = config["mom"], weight_decay = config["wd"])

    # define learning rate scheduler
    scheduler = None
    scheduler_step = None

    if config["lr_s"] == "StepLR":
        # we multiply learning rate with gamma every step_size number of steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
        scheduler_step = "every epoch"
    elif config["lr_s"] == "Cosine":
        # we start with sampled learning rate and decay it to eta_min (sampled lr > eta_min !)
        eta_min = config["lr"] * 1e-3
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epochs, eta_min = eta_min)
        scheduler_step = "every epoch"

    # define criterion to compute loss
    if config["criterion"] == "BCE":
        criterion = nn.BCEWithLogitsLoss()
        # target_type logits --> PyTorch softmaxes within its code and expects raw logits as input
        
    return (model, optimizer, scheduler, scheduler_step, criterion, device)


def train_epoch(dataloader, model, optimizer, scheduler, scheduler_step, criterion, device):
    
    model.train()
    losses = []
    idxs = torch.Tensor([])
    lbls = torch.Tensor([])
    preds = torch.Tensor([])
    
    for batch_idx, batch in enumerate(dataloader):
        # decompose batch and move to device
        idx_batch, img_batch, lbl_batch = batch
        idxs = torch.cat((idxs, idx_batch))
        lbls = torch.cat((lbls, lbl_batch))
        lbl_batch = lbl_batch.type(torch.float32) # cast to long to be able to compute loss
        img_batch, lbl_batch = img_batch.to(device), lbl_batch.to(device)
    
        # zero optimizer gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        logits = model(img_batch).float().squeeze(1)
        loss = criterion(logits.to(device), lbl_batch)
        loss.backward()
        optimizer.step()
        
        # logging
        losses.append(loss.item())
        preds = torch.cat((preds, torch.sigmoid(logits).cpu()))
        
        # schedule learning rate if necessary
        if scheduler_step == "every training batch":
            scheduler.step()
        
    # compute stats
    acc = accuracy_score(lbls.detach().numpy(), (preds.detach().numpy() > 0.5))
    auc = roc_auc_score(lbls.detach().numpy(), preds.detach().numpy())
    loss_mean = np.mean(losses)
    
    return acc, auc, loss_mean


@torch.no_grad()
def test_epoch(dataloader, model, criterion, device):
    
    model.eval()
    losses = []
    idxs = torch.Tensor([])
    lbls = torch.Tensor([])
    preds = torch.Tensor([])
    
    for batch_idx, batch in enumerate(dataloader):
        # decompose batch and move to device
        idx_batch, img_batch, lbl_batch = batch
        idxs = torch.cat((idxs, idx_batch))
        lbls = torch.cat((lbls, lbl_batch))
        lbl_batch = lbl_batch.type(torch.float32) # cast to long to be able to compute loss
        img_batch, lbl_batch = img_batch.to(device), lbl_batch.to(device)
        
        # forward
        logits = model(img_batch).float().squeeze(1)
        loss = criterion(logits.to(device), lbl_batch)
        
        # logging
        losses.append(loss.item())
        preds = torch.cat((preds, torch.sigmoid(logits).cpu()))
        
    # compute stats
    acc = accuracy_score(lbls.detach().numpy(), (preds.detach().numpy() > 0.5))
    auc = roc_auc_score(lbls.detach().numpy(), preds.detach().numpy())
    loss_mean = np.mean(losses)
    
    return acc, auc, loss_mean


def get_metric_from_matrix(matrix, dict_idxs):
    """
    Returns mean of maxs of columns
    """
    result = []
    for k, epoch in dict_idxs.items():
        result.append(matrix[epoch, k])
    return np.mean(result)


def train_evaluate_model(config, trnloaders, valloaders, tstloader, verbose = True, ray = False, return_obj = True, use_model = None):
    """
    Function that aggregates everything in one place to start model training.
    """
    # maximum number of epochs to train
    max_epochs = int(config["max_epochs"])

    # train and evaluate the model
    trn_losses = np.zeros((max_epochs, config["k_cv"]))
    val_losses = np.zeros((max_epochs, config["k_cv"]))
    trn_accs = np.zeros((max_epochs, config["k_cv"]))
    val_accs = np.zeros((max_epochs, config["k_cv"]))
    trn_aucs = np.zeros((max_epochs, config["k_cv"]))
    val_aucs = np.zeros((max_epochs, config["k_cv"]))

    best_models_indicies = {}
    print(f'We have started training') if verbose else None

    # iterate over all train and testloaders (to handle kfolds training)
    for k, (trnloader, valloader) in enumerate(zip(trnloaders, valloaders)):
        
        print(f"Starting fold [{k+1}/{config['k_cv']}]") if verbose else None

        # set up model
        (model, optimizer, scheduler, scheduler_step, criterion, device) = init_training(config)

        # perform training
        max_val_auc = 0

        for epoch in range(max_epochs):

            ##TRAINING##
            trn_acc, trn_auc, trn_loss = train_epoch(trnloader, model, optimizer, 
                                                     scheduler, scheduler_step,
                                                     criterion, device)

            ##VALIDATION##
            val_acc, val_auc, val_loss = test_epoch(valloader, model, criterion, device)

            ##SCHEDULE learning rate if necessary##
            if scheduler_step == "every epoch":
                scheduler.step()

            ##LOGGING##
            trn_losses[epoch, k] = trn_loss
            val_losses[epoch, k] = val_loss
            trn_accs[epoch, k] = trn_acc
            val_accs[epoch, k] = val_acc
            trn_aucs[epoch, k] = trn_auc
            val_aucs[epoch, k] = val_auc
            
            ##REPORT##
            if verbose:
                print(f"Fold [{k+1}/{config['k_cv']}], Epoch [{epoch + 1}/{max_epochs}] --> Trn Loss: {round(trn_loss, 2)}, \
Val Loss: {round(val_loss, 2)}, Trn AUC: {round(trn_auc, 3)}, Val AUC: {round(val_auc, 3)}")
            
            if config["save"]:
                if val_auc >= max_val_auc:
                    max_val_auc = val_auc
                    path = f"best_{config['model']}_fold{k}.pt"
                    torch.save(model.state_dict(), path)
                    best_models_indicies[k] = epoch # needed as we are reporting multiple metrics
    
    if ray:
        session.report({"trn_loss": get_metric_from_matrix(trn_losses, best_models_indicies), "val_loss": get_metric_from_matrix(val_losses, best_models_indicies),
                        "trn_acc": get_metric_from_matrix(trn_accs, best_models_indicies), "val_acc": get_metric_from_matrix(val_accs, best_models_indicies),
                        "trn_auc": get_metric_from_matrix(trn_aucs, best_models_indicies), "val_auc": get_metric_from_matrix(val_aucs, best_models_indicies)})
    
    if len(tstloader) > 0:
        tst_acc, tst_auc, tst_loss = test_epoch(tstloader, model, criterion, device)
        print(f"Test Performance -> Loss: {round(tst_loss, 2)}, AUC: {round(tst_auc, 3)}")

    if return_obj:
        return model, trn_losses, val_losses, trn_accs, val_accs, trn_aucs, val_aucs
    

def ray_trainable(config):
    """
    Function that wraps everything into one function to allow for raytune hyperparameter training.
    """
    # ensure reproducibility (for meaningful hyperparameter selection that does not depend on random seed behavior)
    set_reproducible()

    # initialise objects for training
    trnloaders, valloaders, tstloader = init_data(config)
    
    # perform training (no return!)
    train_evaluate_model(config, trnloaders, valloaders, tstloader, verbose = config["verbose"], ray = True, return_obj = False)


def trial_str_creator(trial):
    """
    Trial name creator for ray tune logging.
    """
    return f"{trial.trial_id}"


def run_ray_experiment(train_func, config, ray_path, num_samples, metric_columns, parameter_columns):

    reporter = JupyterNotebookReporter(
        metric_columns = metric_columns,
        parameter_columns= parameter_columns,
        max_column_length = 15,
        max_progress_rows = 20,
        max_report_frequency = 1, # refresh output table every second
        print_intermediate_tables = True
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"CPU": 16, "GPU": 1}
        ),
        tune_config = tune.TuneConfig(
            metric = "trn_loss",
            mode = "min",
            num_samples = num_samples,
            trial_name_creator = trial_str_creator,
            trial_dirname_creator = trial_str_creator,
            ),
        run_config = air.RunConfig(
            local_dir = ray_path,
            progress_reporter = reporter,
            verbose = 1),
        param_space = config
    )

    result_grid = tuner.fit()
    
    return result_grid


def open_validate_ray_experiment(experiment_path, trainable):
    # open & read experiment folder
    print(f"Loading results from {experiment_path}...")
    restored_tuner = tune.Tuner.restore(experiment_path, trainable = trainable, resume_unfinished = False)
    result_grid = restored_tuner.get_results()
    print("Done!\n")

    # Check if there have been errors
    if result_grid.errors:
        print(f"At least one of the {len(result_grid)} trials failed!")
    else:
        print(f"No errors! Number of terminated trials: {len(result_grid)}")
        
    return restored_tuner, result_grid