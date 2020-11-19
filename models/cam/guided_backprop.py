from typing import Optional
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.cam.guided_grad_cam import GuidedGradCAM


class GuidedBackprop(GuidedGradCAM):
    def __init__(self, model: nn.Module, device):
        super().__init__()
        self.model = model
        self.model.eval()
        self.apply_hooks(self.model)
        self.device = device

    @staticmethod
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False

    @staticmethod
    def apply_hooks(model):
        def forward_hook(m, input, output):
            m.saved_fmap = output.detach()
            print(f"output is saved: {output.shape}")

        def backward_hook(m, grad_in, grad_out):
            assert m.saved_fmap is not None, f"No fmap is saved while backprop in {module}"
            mask = torch.where(m.saved_fmap == 0, 0, 1)
            return (F.relu(grad_out[0] * mask),)

        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    @classmethod
    def save_gb(cls, gb: np.array, prefix_no: Optional[int] = None):
        prefix_no = f"{prefix_no}_" if prefix_no is not None else ""
        gb = cls.gb_processing(gb)
        gb = cls.convert_from_np_to_cv2(gb)
        cv2.imwrite(f"./files/output/images/{prefix_no}gb.jpg", gb)

    def __call__(self, img: torch.Tensor, cls_idx: int) -> np.array:
        img.requires_grad = True
        output = self.model(img)
        mask = torch.zeros(size=output.shape)
        mask[:, cls_idx] = 1
        output = torch.sum(mask * output)
        self.model.zero_grad()
        output.backward()
        gb = img.grad.cpu().data.numpy()[0, :, :, :]
        return gb



