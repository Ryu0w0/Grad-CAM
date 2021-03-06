from typing import Callable
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad as torch_grad
from models.cam.guided_grad_cam import GuidedGradCAM


class GradCAM(GuidedGradCAM):
    def __init__(self, model: nn.Module, f_get_last_module: Callable, device: str):
        """
        Calculate Grad-CAM and produce heatmap.

        Model: Any CNN as a classifier
        f_get_last_module: Function receiving model and returning a module of the last conv layer.
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.f_get_last_module = f_get_last_module
        self.set_forward_hook(model, f_get_last_module)
        self.device = device

    def set_forward_hook(self, model: nn.Module, f_get_last_module):
        """ Register forward hook to store output of the last conv layer """
        def keep_output(self, _, output):
            self.output = output
        f_get_last_module(model).register_forward_hook(keep_output)

    def __call__(self, img: torch.Tensor, cls_idx: int) -> (np.array, np.array, np.array):
        """
        img: input image used for making prediction
        cls_idx: class index visualized to show which region is focused by a model given

        return: heatmap, normalized image
        """
        # get output and feature map from CNN
        output = self.model(img)
        feature_maps = self.f_get_last_module(self.model).output

        # set zeros except index of target class
        output[:, [i for i in range(output.shape[1]) if i != cls_idx]] = 0

        # calc partial derivative of output w.r.t each cell of feature maps
        gradients = torch_grad(outputs=output, inputs=feature_maps,
                               grad_outputs=torch.ones(output.size()).to(device=self.device),
                               create_graph=False, retain_graph=True)[0]

        # calc sum per channel
        alpha = F.avg_pool2d(gradients, feature_maps.shape[2])

        # create localization map
        heatmap = F.relu(torch.sum(feature_maps * alpha, dim=1, keepdim=True))
        heatmap = cv2.resize(heatmap[0, 0, :, :].detach().cpu().numpy(), (299, 299))
        # rescale to [0, 1]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = np.expand_dims(heatmap, axis=0)
        # calc prob on outputs
        probs = F.softmax(output.detach(), dim=1).cpu().numpy()

        return heatmap, probs

