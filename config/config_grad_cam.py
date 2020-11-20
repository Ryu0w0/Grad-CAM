import environ
from utils import logger


@environ.config()
class ConfigGradCAM:
    """
    Config of the scripts.
    """
    # BASIC PARAMETERS
    save_key: str = environ.var(name="SAVE_KEY", converter=str, default="default",
                                help="Used as a log file name.")
    log_level: int = environ.var(name="LOG_LEVEL", converter=logger.get_log_level_from_name,
                                 default="INFO")
    use_gpu: int = environ.var(name="USE_GPU", converter=bool, default=False,
                               help="1 if use GPU otherwise 0")
    # DATASET and TARGET of VISUALIZATION
    wnid: str = environ.var(name="WNID", default="n02113799",
                            help="Category Id of ImageNet for downloading. It should be a specific class of ImageNet.")
    target_cls_idx: int = environ.var(name="TARGET_CLS_INDEX", converter=int, default=267,
                                      help="Specify which index of GT class corresponds to wnid.")
