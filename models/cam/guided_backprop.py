import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.cam.guided_grad_cam import GuidedGradCAM


class GuidedBackprop(GuidedGradCAM):
    def __init__(self, model: nn.Module):
        """
        Compute Guided backpropagation.
        model: any CNN
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.apply_hooks(self.model)

    @staticmethod
    def apply_hooks(model: nn.Module):
        """
        Register forward hook storing output of feature map at each ReLU layer.
        Register backward hook masking values of grads with 0 if output of ReLU
        while forward or output of grads at ReLU is 0.
        """
        def forward_hook(m, input, output):
            m.saved_fmap = output.detach()

        def backward_hook(m, grad_in, grad_out):
            assert m.saved_fmap is not None, f"No fmap is saved while backprop in {module}"
            mask = torch.where(m.saved_fmap == 0, 0, 1)
            return (F.relu(grad_out[0] * mask),)

        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def __call__(self, img: torch.Tensor, cls_idx: int) -> np.array:
        """ Calculate Guided Backpropagation. """
        # Set img as parameters
        img.requires_grad = True
        # Obtain output of the model
        output = self.model(img)
        # Mask output except a target class of output for visualization
        mask = torch.zeros(size=output.shape)
        mask[:, cls_idx] = 1
        output = torch.sum(mask * output)
        # Compute grads of image w.r.t masked output
        self.model.zero_grad()
        output.backward()
        gb = img.grad.cpu().data.numpy()[0, :, :, :]
        return gb




