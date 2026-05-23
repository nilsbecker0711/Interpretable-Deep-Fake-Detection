"""
Contains:

- Simple ViT
- Simple ViT-C (i.e. ViT with a convolutional stem)
- B-cos

Code taken from lucidrain's vit-pytorch:
https://github.com/lucidrains/vit-pytorch/blob/b3e90a265284ba4df00e19fe7a1fd97ba3e3c113/vit_pytorch/simple_vit.py

Paper references
----------------
- Simple ViT: https://arxiv.org/abs/2205.01580
- Simple ViT-C: https://arxiv.org/abs/2106.14881
- B-cos: https://arxiv.org/abs/2205.10268

Note
----
This is compatible with both a non-B-cos SimpleViT and a B-cos SimpleViT,
provided that the correct arguments are passed.

Warning
-------
It is strongly recommended to use the entrypoints defined from `bcos.models.pretrained`
or the `torch.hub` interface to load models, instead of using this directly.
Especially for B-cos models, as they require a LogitBias module at the end of the model,
which the entrypoints below do not include.
Feel free to open up an issue at https://github.com/B-cos/B-cos-v2 if you have any questions.
"""
from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Union
import math
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from bcos.modules import DetachableGNLayerNorm2d, BcosConv2d, LogitLayer, norms, BcosLinear
from bcos.modules.common import DetachableModule
from bcos.common import BcosUtilMixin
from metrics.registry import BACKBONE


__all__ = [
    "SimpleViT",
    # entrypoints
    "vitc_ti_patch1_14",
    "vitc_s_patch1_14",
    "vitc_b_patch1_14",
    "vitc_l_patch1_14",
    "simple_vit_ti_patch16_224",
    "simple_vit_s_patch16_224",
    "simple_vit_b_patch16_224",
    "simple_vit_l_patch16_224",
]
# helpers


def exists(x: Any) -> bool:
    return x is not None


def pair(t: Any) -> Tuple[Any, Any]:
    return t if isinstance(t, tuple) else (t, t)

# classes
class PosEmbSinCos2d(nn.Module):
    def __init__(self, temperature: Union[int, float] = 10_000):
        super().__init__()
        self.temperature = temperature

    def forward(self, patches: Tensor) -> Tensor:
        h, w, dim = patches.shape[-3:]
        device = patches.device
        dtype = patches.dtype

        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
        omega = 1.0 / (self.temperature**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = None,
        b=1.25,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"

        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                norm=norm_layer(dim),
                linear1=linear_layer(dim, hidden_dim, b=b),
                act=act_layer(),
                linear2=linear_layer(hidden_dim, dim, b=b),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(DetachableModule):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        linear_layer: nn.Module = None,
        norm_layer: nn.Module = None,
        b=1.25,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"

        super().__init__()
        self.att = None

        n_lins = 3
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = norm_layer(dim)
        self.pos_info = None
        self.attention_biases = None

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * n_lins, bias=False)
        self.to_out = linear_layer(inner_dim, dim, bias=False, b=b)

    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)
        # x= x.unsqueeze(-1)
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.detach:  # detach dynamic linear weights
            q = q.detach()
            k = k.detach()
            # these are used for dynamic linear w (`attn` below)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = None,
        b=1.25,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"

        super().__init__()

        self.attn = Attention(
            dim,
            heads=heads,
            dim_head=dim_head,
            linear_layer=linear_layer,
            norm_layer=norm_layer,
            b=b,
        )

        self.ff = FeedForward(
            dim,
            mlp_dim,
            linear_layer=linear_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            b=b,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Transformer(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = None,
        b=1.25,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"

        layers_odict = OrderedDict()
        for i in range(depth):
            layers_odict[f"encoder_{i}"] = Encoder(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                linear_layer=linear_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                b=b,
            )
        super().__init__(layers_odict)

@BACKBONE.register_module(module_name="vit_bcos")
class SimpleViT(BcosUtilMixin, nn.Module):
    def __init__(self, vit_config):
        #_warn_if_not_called_from_bcos_models_pretrained_or_torch_hub()
        super(SimpleViT, self).__init__()
        BcosUtilMixin.__init__(self,)
        model_type = vit_config['model_type']
        if model_type == "vitc_ti_patch1_14":
            image_size, patch_size, depth, dim = 14, 1, 11, 384//2
            heads, mlp_dim, conv_stem = 6//2, 1536//2, [24, 48, 96, 192]
        elif model_type == "vitc_s_patch1_14":
            image_size, patch_size, depth, dim = 14, 1, 11, 384
            heads, mlp_dim, conv_stem = 6, 1536, [48, 96, 192, 384]
        elif model_type == "vitc_b_patch1_14":
            image_size, patch_size, depth, dim = 14, 1, 11, 384 * 2
            heads, mlp_dim, conv_stem = 6 * 2, 1536 * 2, [64, 128, 128, 256, 256, 512]
        elif model_type == "vitc_l_patch1_14":
            image_size, patch_size, depth, dim = 14, 1, 13, 1024
            heads, mlp_dim, conv_stem = 16, 1024 * 4, [64, 128, 128, 256, 256, 512]
        elif model_type == "simple_vit_ti_patch16_224":
            image_size, patch_size, depth, dim = 224, 16, 12, 384 // 2
            heads, mlp_dim, conv_stem = 6 // 2, 1536 // 2, None
        elif model_type == "simple_vit_s_patch16_224":
            image_size, patch_size, depth, dim = 224, 16, 12, 384
            heads, mlp_dim, conv_stem = 6, 1536, None
        elif model_type == "simple_vit_b_patch16_224":
            image_size, patch_size, depth, dim = 224, 16, 12, 384 * 2
            heads, mlp_dim, conv_stem = 6 * 2, 1536 * 2, None
        elif model_type == "simple_vit_l_patch16_224":
            image_size, patch_size, depth, dim = 224, 16, 14, 1024
            heads, mlp_dim, conv_stem = 16, 1024 * 4, None
        else:
            image_size=vit_config['image_size']
            patch_size=vit_config['patch_size']
            depth=vit_config['depth'] # Early convs. help transformers see better: reduce depth to account for conv stem for fairness
            dim=vit_config['dim']
            heads=vit_config['heads']
            mlp_dim=vit_config['mlp_dim']
            conv_stem=vit_config['conv_stem'] #[24, 48, 96, 192]

        num_classes = vit_config['num_classes']
        # dim = 384 // 2 #vit_config['dim']
        # depth = 12 #vit_config['depth']
        # heads = vit_config['heads']
        # mlp_dim = vit_config['mlp_dim']
        channels = vit_config.get('channels', 6)  # Default to 6 if not provided
        b = vit_config['b']
        max_out = vit_config.get('max_out', None)
        # image_size = vit_config['image_size']
        # patch_size = vit_config['patch_size']

        #_ = kwargs  # Ignore additional experiment parameters...
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        linear_layer = BcosLinear #(b=b, max_out=max_out) # in_features: int, out_features: int, bias: bool = False, device=None, dtype=None,
        conv2d_layer = BcosConv2d #(b=b, max_out=1) # vit_config.get('conv2d_layer', None)

        # norm_layer = norms.NoBias(norms.DetachableLayerNorm) #vit_config.get('norm_layer', None)
        # norm2d_layer = norms.NoBias(DetachableGNLayerNorm2d) #vit_config.get('norm2d_layer', None)

        # Define a mapping of norm names to norm classes
        norm_mapping = {
            'AllNormUncentered2d': norms.AllNormUncentered2d,
            'BatchNormUncentered2d': norms.BatchNormUncentered2d,
            'GroupNormUncentered2d': norms.GroupNormUncentered2d,
            'GNInstanceNormUncentered2d': norms.GNInstanceNormUncentered2d,
            'GNLayerNormUncentered2d': norms.GNLayerNormUncentered2d,
            'PositionNormUncentered2d': norms.PositionNormUncentered2d,
            'AllNorm2d': norms.AllNorm2d,
            'BatchNorm2d': norms.BatchNorm2d,
            'DetachableGroupNorm2d': norms.DetachableGroupNorm2d,
            'DetachableGNInstanceNorm2d': norms.DetachableGNInstanceNorm2d,
            'DetachableGNLayerNorm2d': norms.DetachableGNLayerNorm2d,
            'DetachableLayerNorm': norms.DetachableLayerNorm,
            'DetachablePositionNorm2d': norms.DetachablePositionNorm2d
        }

        # Retrieve the norm class from the mapping based on config
        norm_class = norm_mapping.get(vit_config['norm'], None)

        if norm_class is None:
            raise ValueError(f"Unknown norm type: {vit_config['norm']}")

        # Apply norm bias if specified in config
        if vit_config.get('norm_bias', False):
            norm_layer = norms.NoBias(norm_class)
        else:
            norm_layer = norm_class

        # norm2d_layer = norm_layer #norms.NoBias(norms.BatchNorm2d)

        # Retrieve the norm class from the mapping based on config
        norm_2d_class = norm_mapping.get(vit_config['norm_2d'], None)

        if norm_2d_class is None:
            raise ValueError(f"Unknown norm type: {vit_config['norm_2d']}")

        # Apply norm bias if specified in config
        if vit_config.get('norm_2d_bias', False):
            norm2d_layer = norms.NoBias(norm_2d_class)
        else:
            norm2d_layer = norm_2d_class


        act_layer = nn.Identity #vit_config.get('act_layer', None)
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"
        
        # conv_stem = vit_config.get('conv_stem', None)  # Default to None if not provided
        if conv_stem:
            assert exists(
                conv2d_layer
            ), "Provide a conv2d layer class when using conv_stem!"
            assert exists(
                norm2d_layer
            ), "Provide a norm2d layer class when using conv_stem!"

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = (
            (channels if conv_stem is None else conv_stem[-1])
            * patch_height
            * patch_width
        )
        stem = (
            dict()
            if conv_stem is None
            else dict(
                conv_stem=make_conv_stem(
                    channels, conv_stem, conv2d_layer, norm2d_layer, act_layer, b=b
                )
            )
        )
        self.to_patch_embedding = nn.Sequential(
            OrderedDict(
                **stem,
                rearrage=Rearrange(
                    "b c (h p1) (w p2) -> b h w (p1 p2 c)",
                    p1=patch_height,
                    p2=patch_width,
                ),
                linear=linear_layer(self.patch_dim, dim, b=b), #max_out=max_out),
            )
        )
        self.positional_embedding = PosEmbSinCos2d()

        dim_head = dim // heads
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            linear_layer=linear_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            b=b,
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            OrderedDict(
                norm=norm_layer(dim),
                linear=linear_layer(dim, num_classes, b=b), #max_out=max_out),
            )
        )
        self.logit_bias = (
            vit_config['logit_bias']
            if vit_config['logit_bias'] is not None
            else math.log(1 / (num_classes - 1))
        )
        # self.logit_temperature = vit_config['logit_temperature']
        # self.logit_layer = LogitLayer(logit_temperature=self.logit_temperature, logit_bias=self.logit_bias)
        #TODO: adapt sequential of model then logit layer BcosSequential(model, self.logit_layer)

    def forward(self, img):
        x = self.features(img)
        out = self.classifier(x)
        return out

    def features(self, inp):
        # print(inp.shape)
        x = self.to_patch_embedding(inp)
        pe = self.positional_embedding(x)
        x = rearrange(x, "b ... d -> b (...) d") + pe

        # print(x.shape)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return x
    
    def classifier(self, x):
        x = self.linear_head(x)
        # x = self.logit_layer(x)
        x = x + self.logit_bias
        return x

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)# or nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # Recursively apply to custom modules
        elif isinstance(module, (BcosConv2d, LogitLayer, BcosLinear, FeedForward, Encoder, nn.Identity)):
            for submodule in module.children():
                self.initialize_weights(submodule)
        # Ignore activation, pooling, and sequential layers
        elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Sequential)):
            pass  # Do nothing
        else:
            print(f'unknown module type {type(module)}')

    def freeze(self):
        # Freeze all layers except the fc layer
        for param in self.parameters():
            param.requires_grad = False  # Freeze all parameters

        for param in self.fc.parameters():
            param.requires_grad = True  # Unfreeze the fc layer



def make_conv_stem(
    in_channels: int,
    out_channels: List[int],
    conv2d_layer: Callable[..., nn.Module] = None,
    norm2d_layer: Callable[..., nn.Module] = None,
    act_layer: Callable[..., nn.Module] = None,
    b=1.25,
):
    """
    Following the conv stem design in Early Convolutions Help Transformers See Better (Xiao et al.)
    """
    model = []
    for outc in out_channels:
        conv = conv2d_layer(
            in_channels,
            outc,
            kernel_size=3,
            stride=(2 if outc > in_channels else 1),
            padding=1,
            b=b,
        )
        in_channels = outc
        norm = norm2d_layer(in_channels)
        act = act_layer()
        model += [conv, norm, act]
    return nn.Sequential(*model)


def _warn_if_not_called_from_bcos_models_pretrained_or_torch_hub():
    """
    Warns the user if this module is not called from bcos.models.pretrained or torch.hub
    """
    import inspect
    import warnings

    # if this file is not called from bcos.models.pretrained or torch.hub, warn the user
    # note: hubconf uses bcos.models.pretrained under the hood
    if not any("pretrained" in call.filename for call in inspect.stack()):
        warnings.warn(
            "You are trying to use the entrypoints from `bcos.models.vit` directly.\n"
            "This is strongly discouraged as it might cause unintended silent errors.\n"
            "Prefer to use the entrypoints from `bcos.models.pretrained` or `torch.hub`.\n"
            f"See lines 17-29 of this file ({__file__}) for why."
        )