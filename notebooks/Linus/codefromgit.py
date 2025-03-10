import torch
import argparse
import os
import pickle
from os.path import join
import numpy as np
import torch.nn.functional as F

# Import utility functions and classes for analysis and explanation methods.
from interpretability.analyses.utils import load_trainer, Analyser
from interpretability.explanation_methods import get_explainer
from interpretability.analyses.localisation_configs import configs

def to_numpy(tensor):
    """
    Converting tensor to numpy.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()

class LocalisationAnalyser(Analyser):
    # Default configuration for this analyser
    default_config = {
        "explainer_name": "Ours",      # Default explainer method name
        "explainer_config": None       # Additional configuration for the explainer
    }
    conf_fn = "conf_results.pkl"       # Filename to store/load computed confidences

    def __init__(self, trainer, config_name, plotting_only=False, verbose=True, **config):
        """
        Initialize the localisation analyser for evaluating the localisation metric (as described in the CoDA-Net paper).
        
        Args:
            trainer: Trainer object used for model predictions and data access.
            config_name: Key to select the specific analysis configuration from 'configs'.
            plotting_only: If True, load previous results (for plotting) rather than re-running analysis.
            verbose: Print warnings if passed parameters are overwritten by config parameters.
            **config: Additional configuration parameters, including:
                - explainer_config: Configuration for the explanation method.
                - explainer_name: Which explanation method to use (default is "Ours").
                - verbose: Whether to display overwrite warnings.
        """
        self.config_name = config_name
        # Retrieve analysis configuration based on config_name.
        analysis_config = configs[config_name]
        if verbose:
            # Warn if any keys in the loaded configuration are overwritten by additional parameters.
            for k in analysis_config:
                if k in config:
                    print("CAVE: Overwriting parameter:", k, analysis_config[k], config[k], flush=True)
        # Update configuration with any additional parameters.
        analysis_config.update(config)
        # Call parent class constructor with the final configuration.
        super().__init__(trainer, **analysis_config)
        if plotting_only:
            # If only plotting, load the stored results and exit initialization.
            self.load_results()
            return
        # Get the explainer method based on the provided explainer name and its configuration.
        self.explainer = get_explainer(trainer, self.config["explainer_name"], self.config["explainer_config"])
        self.sorted_confs = None
        # Create a folder to save results if it doesn't exist.
        save_folder = join(trainer.save_path, self.get_save_folder())
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        # Precompute the sorted confidences from the dataset.
        self.compute_sorted_confs()

    def compute_sorted_confs(self):
        """
        Computes and stores image indices sorted by classifier confidence for each class.
        Loads precomputed confidences if available, otherwise computes them using the test loader.
        """
        # Create the path where results will be stored.
        save_path = join(self.trainer.save_path, "localisation_analysis", "epoch_{}".format(self.trainer.epoch))
        fp = join(save_path, self.conf_fn)

        if os.path.exists(fp):
            print("Loading stored confidences", flush=True)
            with open(fp, "rb") as file:
                self.sorted_confs = pickle.load(file)
            return

        print("No confidences file found, calculating now.", flush=True)
        trainer = self.trainer
        # Initialize a dictionary to store confidences for each class.
        confidences = {i: [] for i in range(trainer.options["num_classes"])}

        # Get the test data loader.
        loader = trainer.data.get_test_loader()
        img_idx = -1
        # Disable gradient computation for efficiency.
        with torch.no_grad():
            # Iterate over test data.
            for img, tgt in loader:
                # Move data to GPU.
                img, tgt = img.cuda(), tgt.cuda()
                # Get the model predictions (logits) and the predicted classes.
                logits, classes = trainer.predict(img, to_probabilities=False).max(1)
                # For each prediction, check if it is correctly classified and store its confidence.
                for logit, pd_class, gt_class in zip(logits, classes, tgt.argmax(1)):
                    img_idx += 1
                    if pd_class != gt_class:
                        continue  # Skip misclassified images.
                    confidences[int(gt_class.item())].append((img_idx, logit.item()))

        # For each class, sort the stored (image index, confidence) pairs in descending order of confidence.
        for k, vlist in confidences.items():
            confidences[k] = sorted(vlist, key=lambda x: x[1], reverse=True)

        # Save the computed confidences to a file for future use.
        with open(fp, "wb") as file:
            pickle.dump(confidences, file)

        self.sorted_confs = confidences

    def get_sorted_indices(self):
        """
        Generates a list of image indices for sampling from the dataset to evaluate multi-image localisation.
        The method selects images by ensuring each block (of n images) contains one image per class, sorted by confidence.
        Returns:
            A list of indices for images sorted by class confidence and sampled randomly.
        """
        idcs = []
        # Get the array of class keys.
        classes = np.array([k for k in self.sorted_confs.keys()])
        # Maintain an index pointer for each class.
        class_indexer = {k: 0 for k in classes}

        # Define a function that returns whether the next available image for a class has a minimum confidence of 50%
        def get_conf_mask_v(_c_idx):
            return torch.tensor(self.sorted_confs[_c_idx][class_indexer[_c_idx]][1]).sigmoid().item() > .5

        # Create a mask for classes that are still confidently classified.
        mask = np.array([get_conf_mask_v(k) for k in classes])
        n_imgs = self.config["n_imgs"]
        # Use a fixed random seed to ensure consistency in class selection.
        np.random.seed(42)
        # Loop until the number of confidently classified classes is less than or equal to the required number of images.
        while mask.sum() > n_imgs:
            # Randomly select n_imgs from the classes that meet the confidence criteria.
            sample = np.random.choice(classes[mask], size=n_imgs, replace=False)

            for c_idx in sample:
                # Get the image index and confidence for the current class.
                img_idx, conf = self.sorted_confs[c_idx][class_indexer[c_idx]]
                # Increment the pointer for this class.
                class_indexer[c_idx] += 1
                # Update the mask: check if the next image for this class is still confidently classified.
                mask[c_idx] = get_conf_mask_v(c_idx) if class_indexer[c_idx] < len(self.sorted_confs[c_idx]) else False
                # Append the selected image index to the output list.
                idcs.append(img_idx)
        return idcs

    def get_save_folder(self, epoch=None):
        """
        Computes the folder path for storing analysis results.
        
        Args:
            epoch: Optionally, the epoch for which to create the folder. Defaults to the current trainer epoch.
        
        Returns:
            The path (string) to the save folder.
        """
        if epoch is None:
            epoch = self.trainer.epoch
        # Folder path includes analysis type, epoch, configuration names, smoothing value, and explainer configuration.
        return join("localisation_analysis", "epoch_{}".format(epoch),
                    self.config_name,
                    self.config["explainer_name"],
                    "smooth-{}".format(int(self.config["smooth"])),
                    self.config["explainer_config"])

    def analysis(self):
        """
        Main analysis function that:
          - Samples multi-images based on the computed sorted indices.
          - Computes attributions using the explainer.
          - Applies smoothing and aggregation of attributions.
          - Computes a localisation metric across samples.
        
        Returns:
            A dictionary with the localisation metric results.
        """
        sample_size, n_imgs = self.config["sample_size"], self.config["n_imgs"]
        trainer = self.trainer
        loader = trainer.data.get_test_loader()
        # Get fixed indices based on class confidence.
        fixed_indices = self.get_sorted_indices()
        metric = []
        explainer = self.explainer
        offset = 0
        # Determine the size of each region by checking one sample image.
        single_shape = loader.dataset[0][0].shape[-1]
        for count in range(sample_size):
            # Create a multi-image and get corresponding targets.
            multi_img, tgts, offset = self.make_multi_image(n_imgs, loader, offset=offset,
                                                            fixed_indices=fixed_indices)
            # Calculate attributions for all classes in the multi-image.
            attributions = explainer.attribute_selection(multi_img, tgts).sum(1, keepdim=True)
            # Apply smoothing if configured.
            if self.config["smooth"]:
                attributions = F.avg_pool2d(attributions, self.config["smooth"], stride=1, padding=(self.config["smooth"] - 1) // 2)
            # Consider only positive attributions.
            attributions = attributions.clamp(0)
            # Compute contribution of attributions per region using average pooling.
            with torch.no_grad():
                contribs = F.avg_pool2d(attributions, single_shape, stride=single_shape).permute(0, 1, 3, 2).reshape(
                    attributions.shape[0], -1)
                total = contribs.sum(1, keepdim=True)
            # Normalize contributions where total > 0.
            contribs = to_numpy(torch.where(total * contribs > 0, contribs/total, torch.zeros_like(contribs)))
            # Save each class's contribution for this sample.
            metric.append([contrib[idx] for idx, contrib in enumerate(contribs)])
            print("{:>6.2f}% of processing complete".format(100*(count+1.)/sample_size), flush=True)
        result = np.array(metric).flatten()
        print("Percentiles of localisation accuracy (25, 50, 75, 100): ", np.percentile(result, [25, 50, 75, 100]))
        return {"localisation_metric": result}

    @staticmethod
    def make_multi_image(n_imgs, loader, offset=0, fixed_indices=None):
        """
        Constructs a multi-image (a composite image) by sampling n_imgs images from the dataset.
        The images are chosen such that each image represents a different class.
        
        Args:
            n_imgs: Number of images to combine (expected to be 4 or 9).
            loader: Data loader for accessing the dataset.
            offset: Current offset index in the dataset to start sampling.
            fixed_indices: Optional list of pre-determined indices (e.g., sorted by confidence).
        
        Returns:
            A tuple (multi_image, targets, new_offset) where:
              - multi_image: The composite image created.
              - targets: List of target labels for each image in the composite.
              - new_offset: Updated offset after sampling.
        """
        assert n_imgs in [4, 9]
        tgts = []  # List to hold target class indices.
        imgs = []  # List to hold individual images.
        count = 0
        i = 0
        # Use fixed indices if provided; otherwise, use the natural order of the dataset.
        if fixed_indices is not None:
            mapper = fixed_indices
        else:
            mapper = list(range(len(loader.dataset)))

        # Loop through the dataset until we have collected n_imgs images of unique classes.
        while count < n_imgs:
            img, tgt = loader.dataset[mapper[i + offset]]
            i += 1
            tgt_idx = tgt.argmax().item()
            # Skip if this class is already included in the multi-image.
            if tgt_idx in tgts:
                continue
            imgs.append(img[None])  # Add an extra dimension to stack later.
            tgts.append(tgt_idx)
            count += 1
        # Concatenate the selected images.
        img = torch.cat(imgs, dim=0)
        # Rearrange images to form a grid (square) multi-image.
        img = img.view(-1, int(np.sqrt(n_imgs)), int(np.sqrt(n_imgs)), *img.shape[-3:]).permute(
            0, 3, 2, 4, 1, 5).reshape(
            -1, img.shape[1], img.shape[2] * int(np.sqrt(n_imgs)), img.shape[3] * int(np.sqrt(n_imgs)))
        # Return the composite image (moved to GPU), target labels, and updated offset.
        return img.cuda(), tgts, i + offset + 1


def argument_parser():
    """
    Create an argument parser for running localisation analysis experiments.
    
    Returns:
        An argparse.ArgumentParser with configured arguments.
    """
    parser = argparse.ArgumentParser(description="Localisation metric analyser.")
    # Define command-line arguments for model checkpoint path, reload options, explainer settings, etc.
    parser.add_argument("--save_path", default=None, help="Path for model checkpoints.")
    parser.add_argument("--reload", default="last",
                        type=str, help="Which epoch to load. Options are 'last', 'best' and 'epoch_X',"
                                       "as long as epoch_X exists.")
    parser.add_argument("--explainer_name", default="Ours",
                        type=str, help="Which explainer method to use. Ours uses trainer.attribute.")
    parser.add_argument("--analysis_config", default="default_3x3",
                        type=str, help="Which analysis configuration file to load.")
    parser.add_argument("--explainer_config", default="default",
                        type=str, help="Which explainer configuration file to load.")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument("--smooth", default=15,
                        type=int, help="Determines by how much the attribution maps are smoothed (avg_pool).")
    return parser


def get_arguments():
    """
    Parse and return command-line arguments.
    
    Returns:
        Parsed command-line options.
    """
    parser = argument_parser()
    opts = parser.parse_args()
    return opts


def main(config):
    # Load the trainer object using the provided model checkpoint path, epoch to reload, and batch size.
    trainer = load_trainer(config.save_path, config.reload, batch_size=config.batch_size)
    # Initialize the localisation analyser with the trainer and configuration parameters.
    analyser = LocalisationAnalyser(trainer, config.analysis_config, 
                                    explainer_name=config.explainer_name,
                                    explainer_config=config.explainer_config, 
                                    smooth=config.smooth)
    # Run the analysis.
    analyser.run()


if __name__ == "__main__":
    # Parse command-line arguments and run the main function.
    params = get_arguments()
    main(params)