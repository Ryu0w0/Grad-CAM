from typing import Optional
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from dataset.img_transform import ImgTransform


class GuidedGradCAM(nn.Module):
    """
    Compute and visualize Guided Grad-CAM based on Grad-CAM and Guided backpropagation.
    """
    @staticmethod
    def freeze_model(model):
        # Unused
        for p in model.parameters():
            p.requires_grad = False

    @classmethod
    def adjust_backprop_values(cls, gb: np.array):
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
    def save_all_output(cls, img: torch.Tensor, heatmap: np.array, ggc: np.array, probs: np.array,
                        target_cls_idx: int, prefix_no: Optional[int] = None):
        """
        Save image of 4 subplots consisted of original image, heatmap, Guided Grad-CAM and
        probability on GT label and predicted label.
        """
        prefix_no = f"{prefix_no}_" if prefix_no is not None else ""

        # Grad-CAM (heatmap)
        img = ImgTransform.denormalize_(img)
        img = img.detach().cpu().numpy()[0, :, :, :]
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.uint8(255 * img)

        heatmap = np.transpose(heatmap, (1, 2, 0))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Guided Grad-CAM
        ggc = cls.adjust_backprop_values(ggc)
        ggc = np.transpose(ggc, (1, 2, 0))
        ggc = np.uint8(255 * ggc)

        # Probability
        prob_gt = probs[0, target_cls_idx]
        predicted_idx = np.argmax(probs, axis=1)[0]
        prob_pred = np.max(probs, axis=1)[0]
        is_tp = "TP" if predicted_idx == target_cls_idx else "FP"

        # PLOT
        fig = plt.figure(tight_layout=True, figsize=(16, 4))
        ax1 = fig.add_subplot(141, title="Original image")
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax2 = fig.add_subplot(142, title="Heatmap")
        ax2.imshow(heatmap)
        ax3 = fig.add_subplot(143, title="Guided Grad-CAM")
        ax3.imshow(ggc)
        ax4 = fig.add_subplot(144, title=f"Probability ({is_tp}) ")
        ax4.bar([0, 1], [prob_gt, prob_pred], tick_label=["GT", "Pred"])

        plt.savefig(f"./files/output/images/{prefix_no}guided_crad_cam.png")

    @staticmethod
    def calc_guided_grad_cam(heatmap: np.array, guided_backprop: np.array) -> np.array:
        return heatmap * guided_backprop
