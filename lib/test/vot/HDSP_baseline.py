import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.HDSP_class import run_vot_exp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

run_vot_exp('HDSP', 'hdsp_deep', vis=False, out_conf=True, channel_type='')
