"""
This script visualizes where a CNN model looks at for prediction by using Grad-CAM.
"""


def initialization():
    """
    Initializing arguments, logger, tensorboard recorder and json files.
    """
    from torch.utils.tensorboard import SummaryWriter
    from config.config_grad_cam import ConfigGradCAM
    from utils import file_operator as f_op
    from utils import logger as log_util
    import environ

    # create config object from arguments
    args = environ.to_config(ConfigGradCAM)

    # create logger
    logger_ = log_util.create_logger("main", "./files/output/logs", args.save_key, args.log_level)
    log_util.logger_ = logger_

    # show specified arguments
    logger_.info("*** ARGUMENTS ***")
    logger_.info(args)

    # create TensorBoard writer
    board_root_dir = f"./files/output/board/{args.save_key}"
    f_op.create_folder(board_root_dir)
    log_util.writer_ = SummaryWriter(board_root_dir)

    # load model config
    logger_.info("** LOAD MODEL CONFIG **")
    config = f_op.load_json("./files/input/models/configs", args.model_config_key)
    logger_.info(config)

    return args, config


def main():
    args, config = initialization()
    import cv2
    import numpy as np
    import torch
    import torchvision
    from torchvision.datasets import ImageFolder
    from torch.nn import functional as F
    from torch.utils.data.dataloader import DataLoader
    from utils.logger import logger_
    from utils import downloader
    import torchvision.models as models
    from dataset.img_transform import ImgTransform
    from models.cam.grad_cam import GradCAM
    from models.cnn.inception import Inception3

    logger_.info("*** SET DEVICE ***")
    device = "cuda" if args.use_gpu else "cpu"
    logger_.info(f"Device is {device}")

    logger_.info("*** CREATE DATASET ***")
    # download images of standard poodles
    # api = downloader.ImageNet(root="./files/input/dataset")
    # api.download("n02113799", "267", verbose=True, limit=32)  # class of standard poodles (idx:267)
    # define transform and dataset
    trans = ImgTransform(args, is_train=False)
    dataset = ImageFolder("./files/input/dataset/img", transform=trans)

    logger_.info("*** CREATE DATA LOADER ***")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    logger_.info("*** LOAD MODEL ***")
    model = Inception3(num_classes=1000, aux_logits=True)
    model.load_pre_train_weights(progress=True)
    model.eval()

    grad_cam = GradCAM(model=model, f_get_last_module=lambda model_: model_.Mixed_7c, device=device)

    logger_.info("*** VISUALIZATION ***")
    for id, batch in enumerate(loader):
        image, _ = batch   # assume batch size is 1
        heatmaps, pred_index, probs = grad_cam(image, cls_idx=267)
        for i in range(len(pred_index)):
            img = trans.denormalize(image)
            img = np.transpose(img.detach().cpu().numpy()[i, :, :, :], (1, 2, 0))
            heatmap = np.transpose(heatmaps[i, :, :, :], (1, 2, 0))
            img = np.uint8(255 * img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            cv2.imwrite(f"./files/output/images/heatmap_{i}.jpg", heatmap)
            cv2.imwrite(f"./files/output/images/img_{i}.jpg", img)

    exit(0)


if __name__ == '__main__':
    main()


