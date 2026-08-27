import os
import sys

#set project path
# training/ must come BEFORE site-packages: the repo ships its own 'bcos' package
# (training/bcos) and an unrelated pip 'bcos' library may also be installed. The
# detectors only sys.path.append(training/), which loses to site-packages.
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAINING_PATH = os.path.join(PROJECT_PATH, "training")
for _p in (PROJECT_PATH, TRAINING_PATH):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import logging
import torch
import argparse
import numpy as np
import pickle
import random
from PIL import Image
from Utils_PointingGame import load_model, load_config, preprocess_image, Analyser
from training.utils.xai.xai_common import canonicalize_grid, adapt_for_model
from training.utils.xai.B_COS_eval import BCOSEvaluator
from training.utils.xai.LIME_eval import LIMEEvaluator
from training.utils.xai.GradCam_eval import GradCamEvaluator
from training.utils.xai.IG_eval import IGEvaluator
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


#######################
# Model/config selection happens via CLI arguments (see parse_args below), e.g.:
#   python GPG_eval.py --model-config training/config/detector/resnet34_bcos_v2.yaml \
#                      --test-config results/test_bcos_res_2_config.yaml \
#                      --weights path/to/ckpt_best.pth --xai-method bcos
#######################

import yaml


def resolve_path(path):
    """Interpret relative paths as relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_PATH, path)


# Creating the SHARED grid assets involves no model at all (--grids-only
# --selection random), so it should not need a detector yaml. These are the only
# values the data loader and the grid writer actually consume in that path; the
# machine-specific ones (rgb_dir, dataset_json_folder, label_dict) still come
# from training/config/test_config.yaml, which is the single place they are
# maintained. Overridable with --set.
GRID_ASSET_DEFAULTS = {
    "mode": "test",
    "lmdb": False,
    "dataset_type": None,        # abstract_dataset.py: 3-channel, model-agnostic
    "resolution": 256,           # every model runs at 256 (unified 2026-08)
    "test_dataset": ["FaceForensics++"],
    "val_dataset": ["FaceForensics++"],
    "frame_num": {"train": 32, "test": 32, "val": 32},
    "compression": "c23",
    "test_batchSize": 32,
    "val_batchSize": 32,
    "workers": 8,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "with_mask": False,
    "with_landmark": False,
    "use_data_augmentation": False,
    # init_data_aug_method() runs unconditionally in the dataset constructor, so
    # these must exist even though nothing is augmented at test time. Mirrors
    # resnet34/vit/xception.yaml, which all set aug_type: simple.
    "aug_type": "simple",
    "data_aug": {"flip_prob": 0.5, "crop_scale": [0.08, 1.0], "rotate_prob": 0.5,
                 "rotate_limit": [-10, 10], "blur_prob": 0.5, "blur_limit": [3, 7],
                 "brightness_prob": 0.5, "brightness_limit": [-0.1, 0.1],
                 "contrast_limit": [-0.1, 0.1], "quality_lower": 40,
                 "quality_upper": 100},
    # grid writer
    "grid_split": 3,
    "max_grids": 500,
    "overwrite": False,
    "base_output_dir": "results/GPG_assets/shared_random",
    # unused without an XAI pass, but the creator's signature wants them
    "xai_method": None,
    "quantitativ": False,
    "threshold_steps": 0,
}


def parse_cli_overrides(pairs):
    """Turn repeated --set KEY=VALUE flags into a config-override dict."""
    overrides = {}
    for item in pairs:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got: {item!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Grid Pointing Game evaluation")
    parser.add_argument("--model-config", default=None,
                        help="Detector yaml, e.g. training/config/detector/resnet34_bcos_v2.yaml. "
                             "Required for evaluation; optional when only creating shared grid "
                             "assets (--grids-only --selection random), which builds no model.")
    parser.add_argument("--test-config", default=None,
                        help="Run overlay yaml, e.g. results/test_bcos_res_2_config.yaml. "
                             "Defaults to training/config/test_config.yaml when creating shared "
                             "grid assets.")
    parser.add_argument("--weights", default=None,
                        help="Checkpoint .pth to evaluate; overrides 'pretrained' from the yamls")
    parser.add_argument("--xai-method", default=None,
                        choices=["bcos", "gradcam", "xgrad", "grad++", "layergrad", "lime", "ig"],
                        help="Overrides xai_method from the test config")
    parser.add_argument("--output-dir", default=None,
                        help="Overrides base_output_dir from the test config")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="test_batchSize override")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra config overrides (repeatable), e.g. "
                             "--set dataset_json_folder=preprocessing/dataset_json_v3")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="Data split to build/evaluate grids from. Use 'val' for the "
                             "fixed monitoring grids consumed during training — the test "
                             "split must stay untouched until the final evaluation.")
    parser.add_argument("--selection", choices=["confidence", "random"], default="confidence",
                        help="confidence = original B-cos protocol (per-model, most-confident "
                             "correct images; needs --weights). random = model-free seeded "
                             "sampling for the fixed SHARED grid sets.")
    parser.add_argument("--grids-only", action="store_true",
                        help="Only create grids, skip the XAI evaluation. With "
                             "--selection random no model/weights are needed at all.")
    parser.add_argument("--seed", type=int, default=32,
                        help="Seed for grid creation (image draw + fake placement). The same "
                             "seed selects the same source images at any resolution.")
    parser.add_argument("--grid-dir", default=None,
                        help="Evaluate an existing (shared) grid folder instead of creating "
                             "grids — e.g. the fixed random assets used by all models.")
    parser.add_argument("--real-selection", choices=["confident", "shuffle"], default="confident",
                        help="How the REAL cells of a grid are drawn. 'confident' = "
                             "most-confident-first, which is what B-cos-v2 does for every "
                             "cell. 'shuffle' = random draw from the confidence-filtered "
                             "pool, the previous behaviour. IGNORED for --selection random, "
                             "which always shuffles so shared grid assets regenerate "
                             "bit-identically.")
    parser.add_argument("--dataset-mixing", choices=["single", "mixed"], default="mixed",
                        help="With several entries in test_dataset: 'single' keeps every "
                             "cell of a grid within ONE dataset (different grids use "
                             "different datasets) so no cell is identifiable by domain "
                             "signature; 'mixed' draws cells from the pooled ranking "
                             "(harder, less saturated, but compression/colour cues "
                             "become a possible shortcut). No effect with one dataset.")
    return parser.parse_args()



#setpup logginglogging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class _LazyGrids:
    """Grid tensors loaded and preprocessed one at a time, on demand.

    The eager version built the whole list on the GPU before evaluating any of
    it (`torch.load(..., map_location=device)` inside a list comprehension). A
    3x3 grid of 256px cells is 768x768x3 float32 = 7.1 MB, and adapt_for_model
    DOUBLES that for b-cos models, which take [x, 1-x] as 6 channels. At the
    ~1500 grids a 100-per-dataset run produces that is 22 GB of VRAM before the
    model even runs: every standard (3ch) run fitted in a 24 GB card at ~10 GB
    and every b-cos run died with CUDA OOM. Loading lazily caps it at one grid.

    Deliberately a sequence, not a generator: the evaluators call len() for
    progress logging and GradCam's layer auto-search slices the first k grids.
    Nothing holds a reference between iterations, so each grid is freed once
    scored.
    """

    def __init__(self, paths, model, mean, std, device):
        self.paths = list(paths)
        self.model = model
        self.mean = mean
        self.std = std
        self.device = device

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        grid = torch.load(path, map_location=self.device)
        grid = canonicalize_grid(grid, mean=self.mean, std=self.std, warn_name=path)
        return adapt_for_model(grid, self.model, mean=self.mean, std=self.std)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._load(p) for p in self.paths[idx]]
        return self._load(self.paths[idx])

    def __iter__(self):
        for path in self.paths:
            yield self._load(path)


class GridPointingGameCreator(Analyser):
    def __init__(self, base_output_dir, grid_size=(3, 3), xai_method=None, max_grids=3,
                 model=None, model_name="default", config_name="default",
                 test_data_loaders=None, dataset=None, device=None, config=None, grid_split=3, overwrite=False, quantitativ=False, threshold_steps=0, b_value_name=0,
                 selection="confidence", grid_dir=None, seed=32, real_selection="confident",
                 datasets=None, dataset_mixing="mixed",
                 topn_fractions=(0.025,), store_images=True):
        """
        selection: "confidence" = original B-cos protocol (model's most-confident
                   correctly-classified images); "random" = model-free seeded
                   random sampling (for the fixed SHARED grid sets used by the
                   in-training monitor and the cross-model comparison).
        grid_dir:  optional override pointing at an existing (shared) grid
                   folder — evaluation only, no ranking pass, no grid creation.
        seed:      RNG seed for grid creation (image draw + fake placement).
        real_selection: how the REAL cells are drawn under selection="confidence".
                   "confident" = most-confident-first (what B-cos-v2 does for every
                   cell); "shuffle" = random draw from the confidence-filtered pool
                   (the previous behaviour, kept so old results can be reproduced).
        datasets:  {name: dataset} for ALL configured test datasets. `dataset` alone
                   is kept for backward compatibility (single-dataset callers such
                   as the in-training monitor).
        dataset_mixing: how cells are drawn when several datasets are present.
                   "single" = every cell of a grid comes from ONE dataset (different
                              grids use different datasets). No within-grid domain
                              cue, so a high score can only mean the manipulation
                              was found.
                   "mixed"  = cells drawn from the pooled multi-dataset ranking. Harder
                              and less saturated, but a cell can stand out by
                              compression/colour signature rather than manipulation —
                              the gap to "single" measures exactly that confound.
        """
        self.grid_size = grid_size
        self.xai_method = xai_method
        self.max_grids = max_grids
        self.model = model
        self.config = config or {}
        self.test_data_loaders = test_data_loaders
        self.dataset = dataset
        # All configured test datasets. Previously only the FIRST was ever used
        # (list(...)[0] in three places), so extra entries in test_dataset were
        # silently ignored.
        self.datasets = datasets or ({"default": dataset} if dataset is not None else {})
        self.dataset_mixing = dataset_mixing
        # store_images=False keeps only the scalar scores. The in-training monitor
        # uses it: a full monitor run wrote ~17 GB of rendered overlays that
        # nothing reads (plot_training_log only opens overall_by_threshold.pkl).
        self.topn_fractions = topn_fractions
        self.store_images = store_images
        self._path_index = None   # {image_path: (dataset_name, idx)}, built lazily
        self._path_datasets = None  # {image_path: [dataset_name, ...]}, built lazily
        self.model_name = model_name
        self.config_name = config_name
        self.device = device
        self.grid_split = grid_split
        self.overwrite = overwrite
        self.quantitativ = quantitativ
        self.threshold_steps = threshold_steps
        self.b_value_name = b_value_name
        self.selection = selection
        self.seed = seed
        self.real_selection = real_selection
        self.output_folder = os.path.join(base_output_dir, f"{model_name}_{config_name}")
        self.confidence_dir= os.path.join(base_output_dir, f"{model_name}_{b_value_name}")
        self.grid_dir = grid_dir or os.path.join(base_output_dir, f"{model_name}_{b_value_name}", f"{grid_size[0]}x{grid_size[1]}")
        self.results_dir = os.path.join(self.output_folder, f"{grid_size[0]}x{grid_size[1]}")

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.confidence_dir, exist_ok=True)
        os.makedirs(self.grid_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Ranking is computed lazily in create_GPG_grids — evaluation of an
        # existing (shared) grid folder needs neither a ranking nor a model pass.
        self.ranking_file = os.path.join(self.confidence_dir, "sorted_confs.pkl")
        self.sorted_confs = None

    def _ensure_ranking(self):
        """Load or compute the image ranking used for grid creation."""
        if self.sorted_confs is not None:
            return
        if os.path.exists(self.ranking_file) and not self.overwrite:
            self.sorted_confs = self.load_ranking(self.ranking_file)
            logger.info("Loaded sorted confidences from %s", self.ranking_file)
            return
        if self.overwrite and os.path.exists(self.ranking_file):
            logger.info("Overwrite is enabled. Recomputing and replacing %s", self.ranking_file)
        if self.selection == "random":
            self.sorted_confs = self.compute_random_ranking()
        else:
            self.sorted_confs = self.compute_sorted_confs()
        self.save_ranking(self.sorted_confs, self.ranking_file)
        logger.info("Saved %s ranking to %s", self.selection, self.ranking_file)

    def _build_path_index(self):
        """{image_path: (dataset_name, idx)} across ALL configured datasets.

        Built once. Replaces the former `self.dataset.image_list.index(path)`
        linear scan, which was both single-dataset and O(n) per lookup.
        """
        if self._path_index is not None:
            return self._path_index
        self._path_index = {}
        for name, ds in self.datasets.items():
            for i, p in enumerate(ds.image_list):
                self._path_index.setdefault(p, (name, i))
        logger.info("Path index: %d images across %d dataset(s): %s",
                    len(self._path_index), len(self.datasets), list(self.datasets))
        return self._path_index

    def dataset_of(self, image_path):
        """Which dataset an image belongs to ('' if unknown).

        First match wins. Fine for FAKES (unique to their subset) and for
        labelling a grid's origin; use datasets_of() when an image may legitimately
        belong to several datasets at once.
        """
        if isinstance(image_path, list) and len(image_path) == 1:
            image_path = image_path[0]
        entry = self._build_path_index().get(image_path)
        return entry[0] if entry else ""

    def datasets_of(self, image_path):
        """EVERY configured dataset containing this image.

        The DF40 '_ff' subsets (FSAll_ff, FRAll_ff, EFSAll_ff, e4e_ff, ...) are
        built on the SAME FaceForensics++ originals, so their real frames are
        literally the same files. dataset_of()'s first-match rule handed all of
        them to whichever subset the index saw first, leaving the others with
        fakes but zero reals — so they could never form a grid and silently
        dropped out of the evaluation. Reals are shared, so every subset that
        contains one gets it in its pool and draws its own sample.
        """
        if isinstance(image_path, list) and len(image_path) == 1:
            image_path = image_path[0]
        if self._path_datasets is None:
            # A SET per path, not a list: the DF40 category datasets (FSAll_ff =
            # 9 face-swap methods, FRAll_ff = 12, EFSAll_ff = 10) list the same
            # real frame once per constituent method, so a plain append repeated
            # the dataset name that many times and a pool ended up holding 9-12
            # copies of every real -- enough for ranked_real[:8] to fill a grid
            # with one image shown eight times.
            acc = {}
            for name, ds in self.datasets.items():
                for p in ds.image_list:
                    acc.setdefault(p, set()).add(name)
            self._path_datasets = {p: sorted(names) for p, names in acc.items()}
            shared = sum(1 for v in self._path_datasets.values() if len(v) > 1)
            logger.info("Path->datasets index: %d images, %d shared by >1 dataset",
                        len(self._path_datasets), shared)
        return self._path_datasets.get(image_path, [])

    def compute_random_ranking(self):
        """Model-free ranking: every image of the split, confidence fixed at 1.0.
        The seeded shuffle in create_GPG_grids then performs the random draw —
        identical for every model, resolution and run (dataset order is stable)."""
        ranking = {0: [], 1: []}
        for name, ds in self.datasets.items():
            for path, label in zip(ds.data_dict['image'], ds.data_dict['label']):
                binary = 0 if label == 0 else 1
                ranking[binary].append((path, 1.0, binary))
        logger.info("Random ranking: %d real, %d fake images across %d dataset(s).",
                    len(ranking[0]), len(ranking[1]), len(self.datasets))
        return ranking

    def compute_sorted_confs(self):
        """Compute ranking by storing (image_path, confidence, label) for each correctly classified image."""
        ranking = {0: [], 1: []}
        # Datasets can OVERLAP: the DF40 '_ff' subsets are all built on the same
        # FaceForensics++ originals, so one real frame is yielded once per subset
        # here. Without this guard it enters the ranking N times, N copies land in
        # the same grid pool, and ranked_real[:8] can fill a grid with eight cells
        # showing the SAME image (measured: 320 real slots drawn from 59 distinct
        # frames, one of them used 34 times). The model output is identical for
        # every copy, so keeping the first is lossless.
        seen = set()
        # Rank over EVERY configured test dataset, not just the first one. The
        # image path carries the dataset identity (see dataset_of), so the
        # ranking tuples stay 3-wide and old pickles remain readable.
        for _ds_name, _loader in self.test_data_loaders.items():
          logger.info("Ranking pass over dataset %s", _ds_name)
          for data_dict in _loader:
            # Move all tensor values in data_dict to the device first.
            for k, value in data_dict.items():
                if value is not None and hasattr(value, 'to'):
                    data_dict[k] = value.to(self.device)
    
            # Now unpack after moving to device.
            img_batch, label_batch, mask, landmark, path_of_image = (
                data_dict[k] for k in ['image', 'label', 'mask', 'landmark', 'image_path']
            )
    
            # Remove extra key if present.
            data_dict.pop('label_spe', None)
            # Convert labels to binary.
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)
    
            num_samples = img_batch.shape[0]
            for j in range(num_samples):
                image = img_batch[j].unsqueeze(0)  # shape: [1, C, H, W]
                label = label_batch[j]
                true_label = int(label.item())
                image_path = path_of_image[j]
                key = image_path[0] if isinstance(image_path, list) and len(image_path) == 1 else image_path
                if key in seen:
                    continue
                seen.add(key)

                # The image comes from the model's OWN dataloader and is already in
                # the model's input space (bcos dataset: [0,1] + inverse channels;
                # standard dataset: mean/std-normalized 3ch). No per-xai-method
                # preprocessing here — the old preprocess_image calls fed 6-channel
                # (or wrongly re-encoded) inputs to 3-channel standard models.
                output = self.model({'image': image, 'label': label})
                logit = output['cls']  # Expected shape: [1, num_classes]

                # Rank by the RAW MAX LOGIT, mirroring B-cos-v2
                # localisation.py:151-156 (`logits, classes = model(img).max(1)`).
                # Their ImageNet baselines are cross-entropy trained as well, so
                # this is the protocol they apply to BOTH model families.
                # Previously we stored a softmax probability instead; in binary
                # that tracks the logit MARGIN, which gives a different ordering.
                max_logit, pred = logit[0].max(0)
                predicted_label = int(pred.item())
                confidence = float(max_logit.item())

                # Only store if prediction is correct
                if true_label == predicted_label:
                    ranking[true_label].append((image_path, confidence, true_label))
    
        # Sort each class's ranking by descending confidence.
        for cls in ranking:
            ranking[cls] = sorted(ranking[cls], key=lambda x: x[1], reverse=True)
            logger.debug("Class %d: %d images after sorting.", cls, len(ranking[cls]))
        return ranking
    
    def get_sorted_image_paths(self):
        """Select top image indices based on rankings for grid creation,
        filtering each tuple by the B-cos-v2 confidence threshold.
        For class 0 (real), selects k * (grid_size[0] * grid_size[1] - 1) images,
        and for class 1 (fake), selects k images.
        """
        # Mirror B-cos-v2 localisation.py:186-192: sigmoid(raw max logit) > 0.5,
        # which is simply logit > 0.
        # NOTE: the previous test (softmax probability > 0.5) could never filter
        # anything. For a correctly classified binary softmax model the predicted
        # class probability is >= 0.5 by construction, and compute_sorted_confs
        # already keeps only correctly classified images.
        def get_conf_mask_v(tup):
            return torch.sigmoid(torch.tensor(float(tup[1]))).item() > 0.5
    
        k = self.max_grids
        # With per-dataset pools the global "top k" cut would starve every dataset
        # but the strongest one (a dataset could keep reals but lose all its fakes,
        # so it can never form a grid). Keep the full confidence-sorted lists and
        # let the per-pool round-robin take most-confident-first within each
        # dataset; the truncation below is only an optimisation for the pooled case.
        per_dataset_pools = self.dataset_mixing == "single" and len(self.datasets) > 1

        sorted_image_paths = {}
        for cls in [0, 1]:
            cls_list = self.sorted_confs.get(cls, [])
            filtered = [tup for tup in cls_list if get_conf_mask_v(tup)]
            logger.debug("Class %d: %d images pass the confidence filter.", cls, len(filtered))
            if per_dataset_pools:
                sorted_image_paths[cls] = filtered
                continue
            required = k * (self.grid_size[0] * self.grid_size[1] - 1) if cls == 0 else k
            sorted_image_paths[cls] = filtered[:required]
        return sorted_image_paths

    def load_sample_by_path(self, image_path, expected_label):
        """
        Retrieve the sample (image, label, etc.) from the dataset by matching the stored image path.
        If the provided image_path is a single-element list, extract the string.
        """
        # If image_path is a list of one element, get the string.
        if isinstance(image_path, list) and len(image_path) == 1:
            image_path = image_path[0]

        # Resolve across ALL datasets via the prebuilt index (was a linear
        # image_list.index() scan on a single dataset).
        entry = self._build_path_index().get(image_path)
        if entry is None:
            raise ValueError(f"Image path {image_path} not found in any dataset.")
        ds_name, idx = entry

        # Retrieve the sample from the dataset using its __getitem__.
        sample = self.datasets[ds_name][idx]  # (image, label, landmark, mask, stored_index)
        sample_label = int(sample[1])
        if sample_label != expected_label:
            raise ValueError(f"Label mismatch at {image_path}: expected {expected_label} but got {sample_label}")
        return sample[0]
    
    def save_ranking(self, ranking, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(ranking, f)

    def load_ranking(self, file_path):
        with open(file_path, "rb") as f:
            ranking = pickle.load(f)
        return ranking
    
    def analysis(self):
        """Evaluate grid tensors and compute overall metrics."""
        raw_results_file = os.path.join(self.results_dir, "results.pkl")

        if os.path.exists(raw_results_file) and not self.overwrite:
            raise RuntimeError(f"Results already exist at {raw_results_file}. Use those results or set overwrite=True.")

        if self.overwrite and os.path.exists(raw_results_file):
            logger.info("Overwrite is enabled. Existing results at %s will be overwritten.", raw_results_file)

        # List grid tensor files.
        grid_paths = [os.path.join(self.grid_dir, f) for f in os.listdir(self.grid_dir) if f.endswith('.pt')]
        logger.info("Found %d grid tensors in %s.", len(grid_paths), self.grid_dir)

        # Load each grid tensor and adapt it to THIS model's input space:
        # canonical [0,1] RGB grids get the model's own preprocessing (standard:
        # mean/std normalization, b-cos: [x, 1-x] channels). Grids stored in a
        # normalized value range are detected and denormalized with a warning.
        mean = self.config.get('mean', [0.5, 0.5, 0.5])
        std = self.config.get('std', [0.5, 0.5, 0.5])
        preprocessed_tensors = _LazyGrids(grid_paths, self.model, mean, std, self.device)
        logger.info("Grids will be loaded on demand (%d total, one at a time).",
                    len(preprocessed_tensors))

        # Choose evaluator based on xai_method.
        if self.xai_method == "bcos":
            evaluator = BCOSEvaluator(self.model, self.device)
        elif self.xai_method == "ig":
            evaluator = IGEvaluator(self.model, self.device)
        elif self.xai_method == "lime":
            evaluator = LIMEEvaluator(self.model, self.device, mean=mean, std=std)
        elif self.xai_method in ["gradcam", "xgrad", "grad++", "layergrad"]:
            evaluator = GradCamEvaluator(self.model, self.device, method=self.xai_method)
        else:
            raise ValueError(f"Unknown xai_method: {self.xai_method}")

        # Run evaluation with thresholding
        raw_results = evaluator.evaluate(
            preprocessed_tensors, grid_paths, self.grid_split,
            threshold_steps=self.threshold_steps,
            topn_fractions=self.topn_fractions, store_images=self.store_images)

        return raw_results

    def create_GPG_grids(self):
        """Create grids by combining ranked real and fake images."""
        # Log and record what ACTUALLY runs, not what was requested:
        # selection="random" forces the real-cell shuffle (see below) and never
        # consults a confidence ranking, so reporting the requested
        # real_selection / a logit ranking key there would be misleading.
        # This single value also drives the shuffle, so the two cannot drift.
        effective_real_selection = (
            "shuffle" if (self.real_selection == "shuffle" or self.selection == "random")
            else "confident")
        ranking_key = ("none (random selection: flat confidence 1.0)"
                       if self.selection == "random"
                       else "raw_max_logit")  # mirrors B-cos-v2 localisation.py

        logger.info("=== Starting GPG grid creation in %s (selection=%s, "
                    "real_selection=%s, ranking_key=%s, seed=%d) ===",
                    self.output_folder, self.selection, effective_real_selection,
                    ranking_key, self.seed)
        random.seed(self.seed)
        self._ensure_ranking()

        manifest = {"selection": self.selection, "seed": self.seed,
                    "real_selection": effective_real_selection,
                    "ranking_key": ranking_key,
                    "model_name": self.model_name, "grid_split": self.grid_split,
                    "grids": []}

        # Check if grids already exist.
        existing_files = [f for f in os.listdir(self.grid_dir) if f.endswith('.pt')]
        logger.debug("Found %d existing .pt files in %s.", len(existing_files), self.grid_dir)

        if existing_files and self.overwrite:
            logger.info("Overwrite is enabled. Deleting existing grid files and continue with creatring new grids.")
            for f in existing_files:
                os.remove(os.path.join(self.grid_dir, f))
            existing_files = []  # Reset list after deletion

        if len(existing_files) >= self.max_grids:
            logger.info("Enough grid files in folder. Skipping grid creation.")
            return
        else:
            for f in existing_files:
                os.remove(os.path.join(self.grid_dir, f))
            existing_files = []  # Reset list after deletion
            

        # Get sorted image paths (tuples of (image_path, confidence, label))
        sorted_image_paths = self.get_sorted_image_paths()
        # Expect one fake (class 1) and remaining real (class 0) images.
        ranked_real = sorted_image_paths.get(0, []).copy()
        ranked_fake = sorted_image_paths.get(1, []).copy()

        # B-cos-v2 fills EVERY cell most-confident-first (localisation.py
        # get_sorted_indices walks down each class's confidence-sorted list).
        # "shuffle" keeps the previous behaviour — a random draw from the
        # confidence-filtered pool — so earlier results remain reproducible.
        #
        # selection="random" ALWAYS shuffles: real_selection is a choice WITHIN
        # the confidence protocol and is meaningless for a model-free draw.
        # Keeping the shuffle here also keeps regeneration of the shared grid
        # assets bit-identical: skipping the call would remove one draw from the
        # RNG stream and shift every later draw, including the shuffle that sets
        # the fake position.
        if effective_real_selection == "shuffle":
            random.shuffle(ranked_real)

        if self.quantitativ:
            random.shuffle(ranked_fake)

        logger.debug("Ranked real: %d, Ranked fake: %d", len(ranked_real), len(ranked_fake))

        # Dataset mixing. "single": every cell of a grid comes from ONE dataset,
        # so no cell can be singled out by compression/colour signature and a high
        # score can only mean the manipulation was localized. "mixed": draw from
        # the pooled ranking — harder and less saturated, but domain cues become a
        # possible shortcut. The gap between the two measures that confound.
        if self.dataset_mixing == "single" and len(self.datasets) > 1:
            # An image goes into the pool of EVERY dataset that contains it, not
            # just the first one (see datasets_of): the DF40 '_ff' subsets share
            # one set of FF++ reals, and first-match attribution starved all but
            # one of them of real cells. ranked_real was shuffled above, so each
            # pool inherits an independent random order and every subset draws
            # its own sample from the shared reals. Pools consume their lists in
            # place, so a shared real may appear in several subsets' grids —
            # intended, they are the same underlying frames — while within one
            # grid the 8 reals stay distinct.
            pools = {}
            for tup in ranked_real:
                for nm in self.datasets_of(tup[0]) or [self.dataset_of(tup[0])]:
                    pools.setdefault(nm, {"real": [], "fake": []})["real"].append(tup)
            for tup in ranked_fake:
                for nm in self.datasets_of(tup[0]) or [self.dataset_of(tup[0])]:
                    pools.setdefault(nm, {"real": [], "fake": []})["fake"].append(tup)
        else:
            pools = {"__all__": {"real": ranked_real, "fake": ranked_fake}}
        pool_names = sorted(pools)
        logger.info("Grid pools (dataset_mixing=%s): %s", self.dataset_mixing,
                    {k: {"real": len(v["real"]), "fake": len(v["fake"])} for k, v in pools.items()})

        n_imgs = self.grid_size[0] * self.grid_size[1]
        logger.debug("Total images per grid: %d", n_imgs)
        side = int(np.sqrt(n_imgs))
        logger.debug("Calculated grid side length: %d", side)
        
        grid_count = 0
        pool_cursor = 0
        while grid_count < self.max_grids:
            logger.info("--- Creating grid %d of %d ---", grid_count + 1, self.max_grids)
            required_real = n_imgs - 1  # Reserve 1 slot for fake image.

            # Round-robin over the pools so dataset coverage is spread evenly
            # instead of exhausting the first dataset before moving on.
            chosen = None
            for _ in range(len(pool_names)):
                nm = pool_names[pool_cursor % len(pool_names)]
                pool_cursor += 1
                if len(pools[nm]["fake"]) >= 1 and len(pools[nm]["real"]) >= required_real:
                    chosen = nm
                    break
            if chosen is None:
                logger.warning("Not enough images left in any pool (need %d real + 1 fake): %s",
                               required_real,
                               {k: {"real": len(v["real"]), "fake": len(v["fake"])} for k, v in pools.items()})
                break
            ranked_real = pools[chosen]["real"]
            ranked_fake = pools[chosen]["fake"]
            logger.debug("Grid from pool %s (real %d, fake %d)", chosen, len(ranked_real), len(ranked_fake))

            # Get first fake tuple and remove it.
            fake_tuple = ranked_fake.pop(0)
            logger.info("Selected fake image: %s with confidence %.4f", fake_tuple[0], fake_tuple[1])
            expected_label = 1
            fake_img = self.load_sample_by_path(fake_tuple[0], expected_label)
            # Grids are stored MODEL-AGNOSTICALLY as canonical raw [0,1] RGB (3ch):
            # bcos samples are sliced to their RGB half, mean/std-normalized samples
            # from the standard dataset are denormalized. Each evaluation then
            # re-applies the target model's own preprocessing (adapt_for_model).
            mean = self.config.get('mean', [0.5, 0.5, 0.5])
            std = self.config.get('std', [0.5, 0.5, 0.5])
            fake_img = canonicalize_grid(fake_img, mean=mean, std=std, warn_name=str(fake_tuple[0]))
            logger.debug("Fake image shape: %s", fake_img.shape if hasattr(fake_img, 'shape') else "N/A")

            # Select first required_real real image tuples.
            selected_real_tuples = ranked_real[:required_real]
            logger.info("Selected real image paths: %s", selected_real_tuples)
            del ranked_real[:required_real]  # consume in place (ranked_real is the pool's list)

            # Retrieve real images using load_sample_by_path for consistency.
            expected_label = 0
            selected_real = [
                canonicalize_grid(self.load_sample_by_path(img_path, expected_label),
                                  mean=mean, std=std, warn_name=str(img_path))
                for img_path, _, _ in selected_real_tuples
            ]
            logger.debug("Retrieved %d real images.", len(selected_real))
            
            # Combine real and fake images.
            images = selected_real + [fake_img]
            logger.debug("Combined image count: %d", len(images))
            random.shuffle(images)  # Shuffle grid placement.
            logger.debug("Images shuffled.")
            
            # Find fake image index in shuffled list.
            fake_index = next(i for i, img in enumerate(images) if torch.equal(img, fake_img))
            final_fake_index = (fake_index % side) * side + (fake_index // side)
            logger.debug("Fake image: shuffled index %d, final index %d", fake_index, final_fake_index)
            
            # Stack images and reshape into grid tensor.
            stacked = torch.stack(images, dim=0)
            logger.debug("Stacked images shape: %s", stacked.shape)
            grid_tensor = (
                stacked.view(-1, side, side, *stacked.shape[-3:])
                       .permute(0, 3, 2, 4, 1, 5)
                       .reshape(-1, stacked.shape[1], stacked.shape[2] * side, stacked.shape[3] * side)
            )
            logger.debug("Grid tensor shape: %s", grid_tensor.shape)
            
            # Save grid tensor with fake position encoded in filename.
            base_name = f"{self.model_name}_{self.b_value_name}_grid_{grid_count}_fake_{final_fake_index}_conf_fake{fake_tuple[1]:.2f}.pt"
            path_to_grid = os.path.join(self.grid_dir, base_name)
            torch.save(grid_tensor, path_to_grid)
            logger.info("Saved grid tensor: %s", path_to_grid)

            manifest["grids"].append({
                "file": base_name,
                "pool": chosen,          # source dataset ('__all__' when mixed)
                "fake_dataset": self.dataset_of(fake_tuple[0]),
                "real_datasets": sorted({self.dataset_of(p) for p, _, _ in selected_real_tuples}),
                "fake_position": int(final_fake_index),
                "fake_image": fake_tuple[0] if not isinstance(fake_tuple[0], list) else fake_tuple[0][0],
                "real_images": [p if not isinstance(p, list) else p[0]
                                for p, _, _ in selected_real_tuples],
            })

            grid_count += 1

        # Provenance record: which images went into which grid, with which seed.
        # The manifest (not the heavy tensors) is what gets committed/compared.
        import json
        with open(os.path.join(self.grid_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)

        logger.info("=== Finished grid creation. Created %d grids (+ manifest.json). ===", grid_count)


def main():
    args = parse_args()

    # Seed BEFORE any dataset construction: the dataset classes shuffle their
    # sample lists at build time with the global RNG (b_cos_pp.py), so without
    # this the grid image selection would differ between runs/resolutions.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model-free path: creating shared random grids needs no model/checkpoint.
    model_free = args.grids_only and args.selection == "random"

    additional_args = {f"{args.split}_batchSize": args.batch_size}
    additional_args.update(parse_cli_overrides(args.set))
    if args.weights is not None:
        additional_args["pretrained"] = resolve_path(args.weights)
    if args.xai_method is not None:
        additional_args["xai_method"] = args.xai_method
    if args.output_dir is not None:
        additional_args["base_output_dir"] = resolve_path(args.output_dir)

    if model_free and args.model_config is None:
        # No detector yaml: built-in defaults + the maintained machine config,
        # so asset generation runs on flags alone.
        model_path = "<GRID_ASSET_DEFAULTS>"
        config_path = resolve_path(args.test_config or "training/config/test_config.yaml")
        config = dict(GRID_ASSET_DEFAULTS)
        with open(config_path, "r") as f:
            config.update(yaml.safe_load(f))
        config.update(additional_args)
        # Assets must land in the same place regardless of the caller's cwd.
        config["base_output_dir"] = resolve_path(config["base_output_dir"])
    else:
        if args.model_config is None or args.test_config is None:
            raise SystemExit("--model-config and --test-config are required unless creating "
                             "shared grid assets with --grids-only --selection random")
        model_path = resolve_path(args.model_config)
        config_path = resolve_path(args.test_config)
        config = load_config(model_path, config_path, additional_args=additional_args)

    # xai_method/quantitativ/threshold_steps only matter once an XAI pass runs.
    required_keys = (["grid_split", "overwrite", "max_grids"] if args.grids_only else
                     ["grid_split", "overwrite", "quantitativ", "xai_method", "max_grids"])
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    config.setdefault("quantitativ", False)
    config.setdefault("threshold_steps", 0)
    config.setdefault("xai_method", None)

    logger.info("Parameters: XAI=%s, Base=%s, Model=%s, Grid=%dx%d, selection=%s",
                config['xai_method'], config['base_output_dir'], model_path,
                config['grid_split'], config['grid_split'], args.selection)

    grid_size = (config['grid_split'], config['grid_split'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_free:
        model = None
        # model-agnostic asset naming: <dataset>_<split>_<resolution>/3x3
        dataset_names = config[f'{args.split}_dataset']
        first_dataset = dataset_names[0] if isinstance(dataset_names, list) else dataset_names
        model_name = f"{first_dataset}_{args.split}"
        b_value_name = str(config["resolution"])
    else:
        model = load_model(config)

        pretrained_path = config['pretrained']
        if not pretrained_path:
            raise ValueError("No checkpoint given: pass --weights or set 'pretrained' in a yaml")
        state_dict = torch.load(pretrained_path)
        # Remove "module." prefix from state_dict keys if necessary.
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        res = model.load_state_dict(new_state_dict, strict=False)
        logger.info("Missing keys: %s", res.missing_keys)
        logger.info("Unexpected keys: %s", res.unexpected_keys)

        model.to(device)
        model.eval()  # Set model to evaluation mode.
        logger.info("Loaded model %s on device %s", model.__class__.__name__, device)

        model_name = config.get("model_name", "defaultModel")
        b_value_name = ("random" if args.selection == "random"
                        else str(config.get("backbone_config", {}).get("b", "default")).replace(".", "_"))

    config_name = os.path.basename(config_path).split('.')[0]

    # Prepare data from the requested split (test for final eval, val for the
    # fixed in-training monitoring grids — keeps the test split untouched).
    # train.py runs argparse at import time — shield our argv from it.
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from train import prepare_testing_data
    finally:
        sys.argv = _argv
    test_data_loaders = prepare_testing_data(config, mode=args.split)
    # Keep EVERY configured dataset. Previously only the first was used, so extra
    # test_dataset entries were silently ignored.
    all_datasets = {name: loader.dataset for name, loader in test_data_loaders.items()}
    dataset = next(iter(all_datasets.values()))  # kept for single-dataset callers
    logger.info("Test datasets in play (%d): %s", len(all_datasets), list(all_datasets))

    # Initialize grid creator with all required objects.
    grid_creator = GridPointingGameCreator(
        base_output_dir=config.get("base_output_dir", "results"),
        grid_size=grid_size,
        xai_method=config["xai_method"],
        max_grids=config["max_grids"],
        model=model,
        model_name=model_name,
        config_name=config_name,
        test_data_loaders=test_data_loaders,
        dataset=dataset,
        device=device,
        config=config,  # needed for mean/std in grid canonicalization/adaptation
        grid_split=config["grid_split"],
        overwrite=config["overwrite"],
        quantitativ=config["quantitativ"],
        threshold_steps=config["threshold_steps"],
        b_value_name=b_value_name,
        selection=args.selection,
        grid_dir=resolve_path(args.grid_dir) if args.grid_dir else None,
        seed=args.seed,
        real_selection=args.real_selection,
        datasets=all_datasets,
        dataset_mixing=args.dataset_mixing,
        # Config-driven so a CLI run can switch the rendered overlays off, the way
        # the in-training monitor already does (trainer.py passes store_images=False).
        # They dominate results_by_threshold.pkl: ~22 MB per grid per threshold
        # sweep, i.e. ~7 GB for a 320-grid, 12-threshold run that nothing reads.
        store_images=config.get("store_images", True),
    )

    if args.grid_dir is None:
        grid_creator.create_GPG_grids()  # Create new grids (skipped for shared assets).
    if not args.grids_only:
        grid_creator.run()               # Run analysis.

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py

