from typing import Optional
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad as torch_grad
from dataset.img_transform import ImgTransform
from models.cam.guided_grad_cam import GuidedGradCAM


class GradCAM(GuidedGradCAM):
    def __init__(self, model: nn.Module, f_get_last_module, device):
        """
        model: model producing outputs used for visualization
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.f_get_last_module = f_get_last_module
        self.device = device
        # self.freeze_model(self.model)
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

    @classmethod
    def save_heatmap(cls, img: torch.Tensor, heatmap: np.array, prefix_no: Optional[int] = None):
        """
        Save heatmap as jpeg.

        heatmap: np.array (W, H)
        img: torch.Tensor (B, C, W, H)
        """
        prefix_no = f"{prefix_no}_" if prefix_no is not None else ""
        img = ImgTransform.denormalize_(img)
        img = cls.convert_from_np_to_cv2(img.detach().cpu().numpy()[0, :, :, :])
        heatmap = cls.convert_from_np_to_cv2(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
        cv2.imwrite(f"./files/output/images/{prefix_no}heatmap.jpg", heatmap)

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
        fmap_size = feature_maps.shape[2]
        alpha = F.avg_pool2d(gradients, fmap_size)

        # create localization map
        heatmap = F.relu(torch.sum(feature_maps * alpha, dim=1, keepdim=True))
        heatmap = cv2.resize(heatmap[0, 0, :, :].detach().cpu().numpy(), (299, 299))
        # rescale to [0, 1]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = np.expand_dims(heatmap, axis=0)

        # get predicted class
        preds, probs = self.__calc_pred_idx(output)

        return heatmap, preds, probs

