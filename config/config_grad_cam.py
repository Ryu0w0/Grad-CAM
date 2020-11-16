import environ
from utils import logger


@environ.config()
class ConfigGradCAM:
    # BASIC PARAMETERS
    save_key: str = environ.var(name="SAVE_KEY", converter=str, default="default",
                                help="Used as a file name of dataset and log files")
    load_model_key: str = environ.var(name="LOAD_MODEL_KEY", converter=str, default="cnn_lr1e-05",
                                      help="Used as a file name of loaded model ($key.ptn)")
    log_level: int = environ.var(name="LOG_LEVEL", converter=logger.get_log_level_from_name,
                                 default="INFO")
    use_gpu: int = environ.var(name="USE_GPU", converter=bool, default=False,
                               help="1 if use GPU otherwise 0")
    use_aug: bool = environ.var(name="USE_AUG", converter=bool, default=False,
                                help="1 if augment train data otherwise 0")
    is_local: bool = environ.var(name="IS_LOCAL", converter=bool, default=True,
                                 help="1 if reduce training data for running with CPU otherwise 0")
    num_folds: bool = environ.var(name="NUM_FOLDS", converter=int, default=5,
                                  help="Specify n for n-folds cross-validation")
    batch_size: bool = environ.var(name="BATCH_SIZE", converter=int, default=1)
    # CNN MODEL
    model_config_key: str = environ.var(name="MODEL_CONFIG_KEY", converter=str, default="cnn_lr1e-05",
                                        help="Name of config file specifying a model architecture.")
    # GRAD CAM
    target_cls_idx: bool = environ.var(name="TARGET_CLS_INDEX", converter=int, default=0,
                                       help="Specify which GT class is visualized by grad-cam")
