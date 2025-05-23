import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR
from .utils import slowfast
#from .vgg19_bcos_detector import VGGBcosDetector
from .resnet34_detector import ResnetDetector
from .v1.resnet34_bcos_detector import ResnetBcosDetector
from .resnet34_bcos_v2_detector import ResnetBcosDetector_v2
from .resnet34_bcos_v2_minimal_detector import ResnetBcosDetector_v2_minimal
from .inception_bcos_detector import InceptionBcosDetector
from .xception_detector import XceptionDetector
#from .vgg19_v2_bcos_detector import VGGBcosDetector
from .vit_bcos_detector import ViTBcosDetector
from .vit_base_detector import ViTBaseDetector
from .convnext_bcos_dector import Convnext_Bcos_Detector
from .vgg19_v2_bcos_detector import VGGBcosDetector
from .xception_bcos_detector import XceptionBcosDetector

""" from .facexray_detector import FaceXrayDetector
from .facexray_detector import FaceXrayDetector
from .facexray_detector import FaceXrayDetector
from .efficientnetb4_detector import EfficientDetector
from .resnet34_detector import ResnetDetector
from .resnet34_bcos_detector import ResnetBcosDetector
from .resnet34_bcos_v2_detector import ResnetBcosDetector_v2
from .f3net_detector import F3netDetector
from .f3net_detector import F3netDetector
from .meso4_detector import Meso4Detector
from .meso4Inception_detector import Meso4InceptionDetector
from .spsl_detector import SpslDetector
from .core_detector import CoreDetector
from .capsule_net_detector import CapsuleNetDetector
from .srm_detector import SRMDetector
from .ucf_detector import UCFDetector
from .recce_detector import RecceDetector
from .fwa_detector import FWADetector
from .ffd_detector import FFDDetector
from .videomae_detector import VideoMAEDetector
from .clip_detector import CLIPDetector
from .timesformer_detector import TimeSformerDetector
from .xclip_detector import XCLIPDetector
from .sbi_detector import SBIDetector
from .ftcn_detector import FTCNDetector
from .i3d_detector import I3DDetector
from .altfreezing_detector import AltFreezingDetector
from .stil_detector import STILDetector
from .lsda_detector import LSDADetector
from .sladd_detector import SLADDXceptionDetector
from .pcl_xception_detector import PCLXceptionDetector
from .iid_detector import IIDDetector
from .lrl_detector import LRLDetector
from .rfm_detector import RFMDetector
from .uia_vit_detector import UIAViTDetector
from .multi_attention_detector import MultiAttentionDetector
from .sia_detector import SIADetector
from .tall_detector import TALLDetector
 """