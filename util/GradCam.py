import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
# import own scripts
import util.inference as inference


class GradCamModel(nn.Module):
    def __init__(self, model, layer_number=4, model_type="ResNet"):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # PRETRAINED MODEL
        self.pretrained = model
        if model_type=="ResNet":
            self.layerhook.append(eval(f"self.pretrained.layer{layer_number}.register_forward_hook(self.forward_hook())"))
        elif model_type=="MobileNet":
            self.layerhook.append(self.pretrained.features[layer_number].register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out


def get_device(num_GPU = 0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{num_GPU}")
    else:
        device = torch.device("cpu")
    return device


def get_heatmap(model, path_to_img, device, model_type="ResNet", layer_number=4, alpha=0.4, against_label=None):
    """
    layer_number indicates which layer to take from model
    against_label if None, then gradients computed against predicted label, if 1 or 0 then
    gradients computed against given label
    """
    # getting image
    img_tensor, img_raw = inference.prep_single_image(path_to_img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    if against_label is None:
        # Getting predictions from model in eval state (since eval turns off droput etc. we
        # need to do it separately
        model.eval()
        # Getting prediction probabilities
        prob = torch.sigmoid(model(img_tensor))
        model.train()
        # Converting probabilities into predictions
        if prob.item() > .5:
            lbl = 1
        else:
            lbl = 0
    else:
        lbl = against_label

    # zeroing gradient
    model.zero_grad()

    # creating GradCam model
    gcmodel = GradCamModel(model=model, layer_number=layer_number, model_type=model_type)

    # get activations and output of model
    gcmodel.eval()
    out, acts = gcmodel(img_tensor)

    # detach activations
    acts = acts.detach().cpu()

    # Convert lbl to a PyTorch tensor
    lbl_tensor = torch.tensor([lbl], dtype=torch.float32).to(device).unsqueeze(0)

    # compute loss
    gcmodel.zero_grad()  # Clear gradients
    loss = nn.BCEWithLogitsLoss()(out, lbl_tensor)
    loss.backward()

    grads = gcmodel.get_act_grads().detach().cpu()

    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    # Creating heatmap
    heatmap_j = torch.mean(acts, dim=1).squeeze()
    heatmap_j_max = heatmap_j.max(axis=0)[0]
    heatmap_j /= heatmap_j_max

    # Upscaling
    heatmap_j_upscaled = F.interpolate(heatmap_j.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear',
                                       align_corners=False)

    # Covnerting colors
    cmap = mpl.cm.get_cmap('jet', 256)
    heatmap_j2 = cmap(heatmap_j_upscaled, alpha=alpha)

    return heatmap_j2, img_raw, lbl


def visualize_heatmap(img, heatmap_j2, lbl, save_fig=False, file_name=None):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.imshow(img, cmap="gray_r")
    if lbl:
        axs.imshow(heatmap_j2[0][0])
    # if lbl == 1:
    #     title = "image having plume"
    # else:
    #     title = "image not having plume"
    # plt.title(f"Gradients areas w.r.t {title}")
    plt.axis('off')
    if save_fig:
        if file_name is not None:
            plt.savefig(file_name + ".png")
            print("Figure saved.")
        else:
            print("Please provide a valid file name for saving the figure.")
    else:
        plt.ioff()
        plt.show()
    return fig
