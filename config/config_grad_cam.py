import environ
from utils import logger


@environ.config()
class ConfigGradCAM:
    # BASIC PARAMETERS
    save_key: str = environ.var(name="SAVE_KEY", converter=str, default="default",
                                help="Used as a file name of dataset and log files")
    log_level: int = environ.var(name="LOG_LEVEL", converter=logger.get_log_level_from_name,
                                 default="INFO")
    use_gpu: int = environ.var(name="USE_GPU", converter=bool, default=False,
                               help="1 if use GPU otherwise 0")
    # DATASET
    wnid: str = environ.var(name="WNID", default="n02113799", help="Category Id of ImageNet for downloading")
    target_cls_idx: int = environ.var(name="TARGET_CLS_INDEX", converter=int, default=267,
                                      help="Specify which index of GT class corresponds to wnid.")
