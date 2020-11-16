import torch
from torch import nn, optim
from utils.seed import seed_everything
from utils import file_operator as f_op
from utils.logger import logger_


class CNN(nn.Module):
    def __init__(self, config: dict, num_class=10):
        super().__init__()
        self.config = config["cnn"]
        self.backbone = nn.Sequential(*self.__build_backbone(config["cnn"]))
        self.classifier = nn.Sequential(*self.__build_classifier(self.backbone, num_class, config["cnn"]))
        self.__initialize_weight()

    @staticmethod
    def __build_backbone(config: dict):
        components = []
        # feature extractor
        for from_, to_, use_pool in config["block_sizes"]:
            components.append(nn.Conv2d(in_channels=from_, out_channels=to_, kernel_size=3, stride=1, padding=1))
            if use_pool:
                components.append(nn.MaxPool2d(kernel_size=2, stride=2))
            components.append(nn.LeakyReLU(0.2))
            components.append(nn.BatchNorm2d(num_features=to_))
        return components

    @staticmethod
    def __build_classifier(backbone: nn.Sequential, num_class: int, config: dict):
        components = list()
        components.append(nn.Flatten())
        num_pooling = len([comp for comp in backbone if isinstance(comp, nn.MaxPool2d)])
        last_resolution = int((config["input_resolution"] / (2 ** num_pooling)))
        last_depth = config["block_sizes"][-1][1]
        components.append(nn.Linear(in_features=last_depth * last_resolution ** 2, out_features=num_class))
        return components

    def __initialize_weight(self):
        for idx, m in enumerate(self.modules()):
            seed_everything(local_seed=idx)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                seed_everything(local_seed=idx)
                nn.init.constant_(m.bias.data, 0)

    def get_optimizer(self):
        lr = self.config["opt"]["lr"]
        beta1, beta2 = self.config["opt"]["betas"]
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                               betas=(beta1, beta2), eps=1e-8)
        return optimizer

    def save(self, save_path: str):
        f_op.create_folder(save_path)
        torch.save(self, f"{save_path}/model.ptn")
        logger_.info(f"Model is saved as {save_path}/model.ptn")

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

