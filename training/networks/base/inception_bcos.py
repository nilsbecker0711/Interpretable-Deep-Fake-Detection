import numpy as np
from collections import namedtuple
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Callable, Any, Optional, Tuple, List
from bcos.bcosconv2d import BcosConv2d
from metrics.registry import BACKBONE
import logging
from bcos.detector_utils import MyAdaptiveAvgPool2d, FinalLayer

logger = logging.getLogger(__name__) 

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

@BACKBONE.register_module(module_name="inception_bcos")
class Inception3(nn.Module):

    def __init__(self, inceptionnet_config) -> None:
        """ Constructor
        Args:
            inceptionnet_config: configuration file with the dict format
        """

        super(Inception3, self).__init__()

        num_classes: int = inceptionnet_config["num_classes"]
        aux_logits: bool = inceptionnet_config["aux_logits"]
        transform_input: bool = inceptionnet_config["transform_input"]
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = inceptionnet_config["inception_blocks"]
        init_weights: Optional[bool] = inceptionnet_config["init_weights"]
        self.log_temperature: int = inceptionnet_config["log_temperature"]
        self.bias: float = np.log(inceptionnet_config["bias"][0]/inceptionnet_config["bias"][1])
        self.b: float = inceptionnet_config["b"]

        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(6, 32, kernel_size=3, stride=2, b=self.b)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3, b=self.b)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1, b=self.b)
        # Diff to torchvision: maxpool -> avgpool
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2)
        # Diff End
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1, b=self.b)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3, b=self.b)
        # Diff to torchvision: maxpool -> avgpool
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=2)
        # Diff End
        self.Mixed_5b = inception_a(192, pool_features=32, b=self.b)
        self.Mixed_5c = inception_a(256, pool_features=64, b=self.b)
        self.Mixed_5d = inception_a(288, pool_features=64, b=self.b)
        self.Mixed_6a = inception_b(288, b=self.b)
        self.Mixed_6b = inception_c(768, channels_7x7=128, b=self.b)
        self.Mixed_6c = inception_c(768, channels_7x7=160, b=self.b)
        self.Mixed_6d = inception_c(768, channels_7x7=160, b=self.b)
        self.Mixed_6e = inception_c(768, channels_7x7=192, b=self.b)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768, b=self.b)
        self.Mixed_7b = inception_e(1280, b=self.b)
        self.Mixed_7c = inception_e(2048, b=self.b)
        # self.dropout = nn.Dropout()
        # Diff to torchvision: no avgpool and linear -> BcosConv2d
        self.fc = BcosConv2d(2048, num_classes, kernel_size=1, stride=1, padding=0, scale_fact=200, b = self.b)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Add global average pooling
        self.flatten = nn.Flatten()                        # Add flatten layer
        self.debug = False
        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            print("Weights have been initialized.")

    def get_features(self, x):
        return self.get_sequential_model()[:-1](x)

    def _transform_input(self, x: Tensor) -> Tensor:
        return x
        # if self.transform_input:
        #     x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        #     x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        #     x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        #     x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # return x

    def get_sequential_model(self):
        """
        For evaluation purposes only, to extract layers at roughly the same relative network depth between
        different models.
        """
        model = nn.Sequential(
            self.Conv2d_1a_3x3,
            self.Conv2d_2a_3x3,
            self.Conv2d_2b_3x3,
            self.avgpool1,
            self.Conv2d_3b_1x1,
            self.Conv2d_4a_3x3,
            self.avgpool2,
            self.Mixed_5b,
            self.Mixed_5c,
            self.Mixed_5d,
            self.Mixed_6a,
            self.Mixed_6b,
            self.Mixed_6c,
            self.Mixed_6d,
            self.Mixed_6e,
            self.Mixed_7a,
            self.Mixed_7b,
            self.Mixed_7c
            #self.fc
        )
        return model

    def get_layer_idx(self, idx):
        """
        For evaluation purposes only, to extract layers at roughly the same relative network depth between
        different models.
        """
        return int(np.ceil(len(self.get_sequential_model())*idx/10))

    def print(self, layer_name, x):
        if self.debug:
            print(layer_name, x.shape)

    def _inception_impl(self, x: Tensor):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # self.print("Conv2d_1a_3x3", x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # self.print("Conv2d_2a_3x3", x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # self.print("Conv2d_2b_3x3", x)
        # N x 64 x 147 x 147
        x = self.avgpool1(x)
        # self.print("avgpool1", x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # self.print("Conv2d_3b_1x1", x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # self.print("Conv2d_4a_3x3", x)
        # N x 192 x 71 x 71
        x = self.avgpool2(x)
        # self.print("avgpool2", x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # self.print("Mixed_5b", x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # self.print("Mixed_5c", x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # self.print("Mixed_5d", x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # self.print("Mixed_6a", x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # self.print("Mixed_6b", x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # self.print("Mixed_6c", x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # self.print("Mixed_6d", x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # self.print("Mixed_6e", x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                self.aux_out = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # self.print("Mixed_7a", x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        #self.print("Mixed_7b", x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        #self.print("Mixed_7c", x)
        # N x 2048 x 8 x 8
        # N x 2048 x 1 x 1
        # x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]):
        if self.training and self.aux_logits:
            return x
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, None)
        else:
            return self.eager_outputs(x, None)

    def features(self, inp):
        "Gets features from backbone"
        x = self._inception_impl(inp)
        return x

    def classifier(self, features):
        '''The prediction head, consisting of gobal avg pool and fc'''
        pooling = MyAdaptiveAvgPool2d((1, 1))
        x = pooling.forward(in_tensor = features)
        final = FinalLayer(bias = self.bias, norm = self.log_temperature)
        x = final.forward(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out


class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        b: float,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        self.b = b
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1, b=self.b)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1, b=self.b)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2, b=self.b)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, b=self.b)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1, b=self.b)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1, b=self.b)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1, b=self.b)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        b: float,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        self.b = b
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2, b=self.b)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, b=self.b)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1, b=self.b)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2, b=self.b)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        # Diff to torchvision: max->avg pool
        branch_pool = self.pool(x)
        # Diff End
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        b: float,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        self.b = b
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1, b=self.b)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1, b=self.b)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), b=self.b)
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0), b=self.b)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1, b=self.b)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), b=self.b)
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), b=self.b)
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), b=self.b)
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3), b=self.b)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1, b=self.b)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        b: float,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        self.b = b
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1, b=self.b)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2, b=self.b)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1, b=self.b)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3), b=self.b)
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0), b=self.b)
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2, b=self.b)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        # Diff to torchvision: max->avg
        branch_pool = self.pool(x)
        # Diff End
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        b: float,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        self.b = b
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1, b=self.b)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1, b=self.b)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), b=self.b)
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), b=self.b)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1, b=self.b)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1, b=self.b)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), b=self.b)
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), b=self.b)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1, b=self.b)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        b: float,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        self.b = b
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1, b=self.b)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = conv_block(128, 768, kernel_size=5, b=self.b)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        # Diff to torchvision: linear -> BcosConv2d
        self.fc = BcosConv2d(768, num_classes, kernel_size=1, stride=1, padding=0, scale_fact=200, b=self.b)
        # Diff End
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = self.pool(x)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        x = self.fc(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))[..., 0, 0]
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        b: float,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        # Diff to torchvision: no batch norm, conv -> proj-conv
        # if isinstance(kwargs["kernel_size"], int):
        #     kwargs["padding"] = (kwargs["kernel_size"] - 1) // 2
        # else:
        #     kwargs["padding"] = tuple((np.array(kwargs["kernel_size"])-1)//2)
        self.conv = BcosConv2d(in_channels, out_channels, scale_fact=200, b=b, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
        # Diff End