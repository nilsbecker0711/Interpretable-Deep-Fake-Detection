import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import BACKBONE

""" 
from .mesonet import Meso4, MesoInception4

from .efficientnetb4 import EfficientNetB4
from .xception_sladd import Xception_SLADD """
from .inception_bcos import Inception3
from .xception import Xception
from .resnet34 import ResNet34
from .resnet34_bcos import ResNet34_bcos
from .resnet34_bcos_v2 import ResNet34_bcos_v2
