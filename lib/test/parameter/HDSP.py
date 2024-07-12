from .lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from vlib.config.vipt.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    yaml_file = os.path.join(prj_dir, 'experiments/HDSP/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    # Network checkpoint path
    params.checkpoint=''
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params