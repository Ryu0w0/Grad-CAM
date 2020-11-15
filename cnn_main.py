"""
This script train CNN with applying Grad-CAM.
"""


def initialization():
    """
    Initializing arguments, logger, tensorboard recorder and json files.
    """
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    from utils import file_operator as f_op
    from utils import logger as log_util
    from utils import seed
    from utils.config import Config
    import environ

    # create config object from arguments
    args = environ.to_config(Config)

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

    # set flg of using seeds
    if args.is_seed:
        seed.feed_seed = True

    return args, config


def main():
    args, config = initialization()
    from utils.logger import logger_
    from dataset.sub_cifar10.cifar10_cv import CIFAR10CV
    from dataset.sub_cifar10.cifar10_test import CIFAR10Test
    from trainer.sub_trainer.train_only_cnn import TrainOnlyCNN

    logger_.info("*** SET DEVICE ***")
    device = "cuda" if args.use_gpu else "cpu"
    logger_.info(f"Device is {device}")

    logger_.info("*** CREATE DATASET ***")
    trainset = CIFAR10CV(root='./files/input/dataset', train=True, download=True, args=args,
                         reg_map=config["train_data_regulation"],
                         expand_map=config["train_data_expansion"] if "train_data_expansion" in config.keys() else None)
    testset = CIFAR10Test(root='./files/input/dataset', train=False, download=True, args=args, cifar10_cv=trainset)

    logger_.info("*** DEFINE TRAINER ***")
    trainer_cv = TrainOnlyCNN(trainset, args, config, device)
    trainer_test = TrainOnlyCNN(testset, args, config, device)

    # training and validation
    if args.do_cv:
        logger_.info("*** CROSS-VALIDATION ***")
        trainer_cv.cross_validation()
    if args.do_test:
        logger_.info("*** TEST ***")
        trainer_test.cross_validation()

    exit(0)


if __name__ == '__main__':
    main()


