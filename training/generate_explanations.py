"""
Generate B-cos explanation heatmaps from a trained model checkpoint.
Saves one PNG per image: [original | class-0 heatmap | class-1 heatmap | class-1 overlay].
Run from the training/ directory.
"""
import os
import sys
import argparse
import yaml
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors import DETECTOR
from utils.explanation_visualizer import save_explanation_grid

# train.py runs argparse at module level, so mask sys.argv before importing
# to prevent it from seeing our arguments and failing.
_real_argv = sys.argv
sys.argv = ["train.py"]
from train import prepare_testing_data
sys.argv = _real_argv


def load_config(detector_yaml, additional_args=None):
    with open(detector_yaml, "r") as f:
        config = yaml.safe_load(f)
    try:
        with open("./config/train_config.yaml", "r") as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser(
            "~/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml"
        ), "r") as f:
            config2 = yaml.safe_load(f)
    if "label_dict" in config:
        config2["label_dict"] = config["label_dict"]
    config.update(config2)
    if additional_args:
        config.update(additional_args)
    return config


def load_model(config, checkpoint_path, device):
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    new_state_dict = OrderedDict(
        (k.replace("module.", ""), v) for k, v in state_dict.items()
    )
    result = model.load_state_dict(new_state_dict, strict=False)
    if result.missing_keys:
        print(f"Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {result.unexpected_keys}")

    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate B-cos explanation heatmaps for a trained model."
    )
    parser.add_argument(
        "--detector_yaml", required=True,
        help="Path to detector config YAML "
             "(e.g. config/detector/resnet34_bcos_v2_minimal.yaml)"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to ckpt_best.pth"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Directory to write output PNGs into"
    )
    parser.add_argument(
        "--num_images", type=int, default=20,
        help="Number of images to explain (default: 20)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for the data loader (default: 4)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(
        args.detector_yaml,
        additional_args={
            "pretrained":     args.checkpoint,
            "test_batchSize": args.batch_size,
        },
    )
    config["mode"] = "test"

    model = load_model(config, args.checkpoint, device)
    print(f"Model: {config['model_name']}")

    test_loaders = prepare_testing_data(config, mode="test")
    loader = list(test_loaders.values())[0]
    print(f"Test set: {len(loader.dataset)} images")

    saved = save_explanation_grid(
        model, loader, args.out_dir, device,
        num_images=args.num_images,
        step=None,   # no step prefix for standalone use
    )
    print(f"\nDone — saved {saved} explanations to {args.out_dir}")


if __name__ == "__main__":
    main()
