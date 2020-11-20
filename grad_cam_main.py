"""
This script visualizes where a CNN model looks at for prediction by using Grad-CAM.
"""


def initialization():
    """
    Initializing arguments, logger, tensorboard recorder and json files.
    """
    from torch.utils.tensorboard import SummaryWriter
    from utils import file_operator as f_op
    from utils import logger as log_util
    import environ
    from config.config_grad_cam import ConfigGradCAM

    # create config object from arguments
    args = environ.to_config(ConfigGradCAM)

    # create logger
    logger_ = log_util.create_logger("main", "./files/output/logs", args.save_key, args.log_level)
    log_util.logger_ = logger_

    # show specified arguments
    logger_.info("*** ARGUMENTS ***")
    logger_.info(args)

    return args


def main():
    from config.config_grad_cam import ConfigGradCAM
    args: ConfigGradCAM = initialization()

    import copy
    import torch
    from torchvision.datasets import ImageFolder
    from torch.utils.data.dataloader import DataLoader
    from utils.logger import logger_
    from dataset.img_transform import ImgTransform
    from utils.downloader import ImageNet
    from models.cam.grad_cam import GradCAM
    from models.cam.guided_backprop import GuidedBackprop
    from models.cam.guided_grad_cam import GuidedGradCAM
    from models.cnn.inception import Inception3

    logger_.info("*** SET DEVICE ***")
    device = "cuda" if args.use_gpu else "cpu"
    logger_.info(f"Device is {device}")

    logger_.info("*** DOWNLOAD DATASET ***")
    # download images of standard poodles
    api = ImageNet(root="./files/input/dataset")
    api.download(str(args.wnid), str(args.target_cls_idx), verbose=True, limit=32)

    logger_.info("*** CREATE DATASET ***")
    trans = ImgTransform(args, is_train=False)
    dataset = ImageFolder("./files/input/dataset/img", transform=trans)

    logger_.info("*** CREATE DATA LOADER ***")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    assert loader.batch_size == 1, f"batch size should be 1 but got {loader.batch_size}"

    logger_.info("*** LOAD CNN MODEL ***")
    model_gc = Inception3(num_classes=1000, aux_logits=True).to(device)
    model_gc.load_pre_train_weights(progress=True)
    model_gp = copy.deepcopy(model_gc).to(device)

    logger_.info("*** Prepare GradCAM and Guided Backprop procedures ***")
    grad_cam = GradCAM(model=model_gc,
                       f_get_last_module=lambda model_: model_.Mixed_7c, device=device)
    guided_backprop = GuidedBackprop(model=model_gp)

    logger_.info("*** VISUALIZATION ***")
    for id, batch in enumerate(loader):
        logger_.info(f"[{id+1}/{len(loader)}] processing...")
        image, _ = batch
        # calc and visualize heatmap of Grad-CAM
        heatmap, probs = grad_cam(image, cls_idx=args.target_cls_idx)
        # calc and visualize guided backprop
        gb = guided_backprop(image, cls_idx=args.target_cls_idx)
        # calc and visualize Guided Grad-CAM
        ggc = GuidedGradCAM.calc_guided_grad_cam(heatmap, gb)
        GuidedGradCAM.save_all_output(image, heatmap, ggc, probs, args.target_cls_idx, id)

    exit(0)


if __name__ == '__main__':
    main()


