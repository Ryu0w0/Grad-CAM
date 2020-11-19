from typing import Optional
import cv2
import numpy as np
import torch
from torch import nn
from dataset.img_transform import ImgTransform


class GuidedGradCAM(nn.Module):
    @staticmethod
    def freeze_model(model):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    @staticmethod
    def convert_from_np_to_cv2(img_array: np.array) -> np.array:
        """
        Convert 3 things below;
         - Channel from (C, W, H) to (W, H, C)
         - Value range from [0, 1] to [0, 255]
         - Order of colors from RGB to BGR

         Input should be (C, W, H) ranged in [0, 1] as RGB/Gray scale
        """
        img_array = np.transpose(img_array, (1, 2, 0))
        img_array = np.uint8(255 * img_array)
        if img_array.shape[2] == 3:  # change color order if color image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array

    @classmethod
    def save_img(cls, img: torch.Tensor, prefix_no: Optional[int] = None):
        """
        Save image as jpeg.
            img: torch.Tensor (B, C, W, H)
            prefix_no: iter_no of data loader
        """
        prefix_no = f"{prefix_no}_" if prefix_no is not None else ""
        img = ImgTransform.denormalize_(img)
        img = cls.convert_from_np_to_cv2(img.detach().cpu().numpy()[0, :, :, :])
        cv2.imwrite(f"./files/output/images/{prefix_no}img.jpg", img)

    @classmethod
    def gb_processing(cls, gb: np.array):
        """
        Retrieved from https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py#L222
        gb: np.array of (C, W, H)
        """
        gb = gb - np.mean(gb)
        gb = gb / (np.std(gb) + 1e-5)
        gb = gb * 0.1
        gb = gb + 0.5
        gb = np.clip(gb, 0, 1)
        return gb

    @classmethod
    def save_guided_grad_cam(cls, ggc: np.array, prefix_no: Optional[int] = None):
        """
        ggc: np.array of (3, W, H)
        """
        prefix_no = f"{prefix_no}_" if prefix_no is not None else ""
        ggc = cls.gb_processing(ggc)
        ggc = cls.convert_from_np_to_cv2(ggc)
        cv2.imwrite(f"./files/output/images/{prefix_no}ggc.jpg", ggc)

    @staticmethod
    def calc_guided_grad_cam(heatmap: np.array, guided_backprop: np.array) -> np.array:
        return heatmap * guided_backprop
