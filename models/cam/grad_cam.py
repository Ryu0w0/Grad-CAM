import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad as torch_grad
from models.cnn.inception import Inception3


class GradCAM(nn.Module):
    def __init__(self, model: nn.Module, f_get_last_module, device):
        """
        model: model producing outputs used for visualization
        """
        super().__init__()
        self.model = model
        self.device = device
        self.f_get_last_module = f_get_last_module
        self.set_hook(model, f_get_last_module)

    def set_hook(self, model, f_get_last_module):
        def keep_output(self, _, output):
            self.output = output
        f_get_last_module(model).register_forward_hook(keep_output)

    @staticmethod
    def __calc_pred_idx(output: torch.Tensor) -> (np.array, np.array):
        output = output.detach().cpu()
        probs = F.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        return predicted.numpy(), probs.numpy()

    def forward(self, img: torch.Tensor, cls_idx: int) -> (np.array, np.array, np.array):
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
        fmap_size = feature_maps.shape[2]
        alpha = F.avg_pool2d(gradients, fmap_size)

        # create localization map
        local_map = F.relu(torch.sum(feature_maps * alpha, dim=1, keepdim=True))
        # rescale to [0, 1]
        local_map = (local_map - torch.min(local_map)) / (torch.max(local_map) - torch.min(local_map))
        local_map = F.interpolate(local_map, (299, 299)).detach().cpu().numpy()

        # get predicted class
        preds, probs = self.__calc_pred_idx(output)

        return local_map, preds, probs

