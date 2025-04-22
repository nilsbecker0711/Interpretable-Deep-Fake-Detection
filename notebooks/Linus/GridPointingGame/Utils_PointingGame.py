import os
import sys
import pickle


#set project path
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

import yaml
import logging
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(config):
    """Load model from the DETECTOR registry using the config."""
    logger.info("Registered models: %s", list(DETECTOR.data.keys()))
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    return model

def load_config(model_path, confiq_path, additional_args={}):
    """Load config from YAML file and merge with test config and additional overrides."""
    with open(model_path, 'r') as f:
        config = yaml.safe_load(f)
    # Try loading test config from local file; otherwise, use home directory.
    with open(confiq_path, 'r') as f:
        config2 = yaml.safe_load(f)
    # Use label dictionary from primary config if available.
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    config.update(config2)
    # Adjust configuration for dry run if specified.
    if config.get('dry_run', False):
        config['nEpochs'] = 0
        config['save_feat'] = False
    # Update with any additional arguments.
    for key, value in additional_args.items():
        config[key] = value
    return config

def preprocess_image(img):
    # If image tensor has 3 channels, add inverse channels (for 6-channel encoding).
    if img.shape[1] == 3:
        img = torch.cat([img, 1.0 - img], dim=1)
    return img

class Analyser:
    def analysis(self):
        raise NotImplementedError("Need to implement analysis function.")

    def run(self):
        # Run analysis and save both raw and overall results.
        overall, raw = self.analysis()
        self.save_results(raw, overall)

    def save_results(self, raw, overall):
        """Save grouped results and overall metrics per threshold."""
        import collections

        # Group raw results by threshold
        threshold_groups = collections.defaultdict(list)
        for entry in raw:
            threshold = entry.get("threshold", None)
            threshold_groups[threshold].append(entry)

        # Save all threshold groups in a single file
        all_raw_path = os.path.join(self.results_dir, "results_by_threshold.pkl")
        with open(all_raw_path, "wb") as f:
            pickle.dump(dict(threshold_groups), f)
        logger.info("Saved all raw results grouped by threshold to %s", all_raw_path)

        # Compute per-threshold overall metrics
        overall_by_threshold = {}
        for threshold, results in threshold_groups.items():
            accuracies = [res["accuracy"] for res in results]
            percentiles = np.percentile(np.array(accuracies), [25, 50, 75, 100])
            overall_by_threshold[threshold] = {
                "localisation_metric": accuracies,
                "percentiles": percentiles
            }

        # Save overall metrics in a single file
        overall_path = os.path.join(self.results_dir, "overall_by_threshold.pkl")
        with open(overall_path, "wb") as f:
            pickle.dump(overall_by_threshold, f)
        logger.info("Saved overall metrics grouped by threshold to %s", overall_path)

    def load_results(self, load_overall=True):
        """Load results from disk and print summary info."""
        file_path = os.path.join(self.results_dir, "overall.pkl" if load_overall else "results.pkl")
        with open(file_path, "rb") as f:
            loaded = pickle.load(f)
        logger.info("Results loaded from %s", file_path)
        if load_overall:
            localisation_metric = loaded.get("localisation_metric", None)
            percentiles = loaded.get("percentiles", None)
            if localisation_metric is not None and percentiles is not None:
                logger.info("Overall results: %d evaluations, percentiles: %s", len(localisation_metric), percentiles)
            else:
                logger.warning("Overall results missing expected keys.")
        else:
            sorted_results = sorted(loaded, key=lambda res: res.get("accuracy", 0), reverse=True)
            logger.info("Top raw results:")
            for idx, res in enumerate(sorted_results[:10]):
                logger.info("[%d] %s - Accuracy: %s", idx + 1, res.get("path", "N/A"), res.get("accuracy", "N/A"))
        return loaded