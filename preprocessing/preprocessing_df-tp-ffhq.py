import os
import json
import random
import argparse

def create_json(real_folder, fake_folder, train_ratio=0.8, dataset_name="df-tp-ffhq"):
    """
    Creates a JSON structure for the new dataset.

    Args:
        real_folder (str): Path to the folder with real images (ffhq_imgs).
        fake_folder (str): Path to the folder with fake images (stylegan2_psi1.0).
        train_ratio (float): Ratio of images to assign to training (default 0.8).
        dataset_name (str): Name for the dataset (default "NewDataset").

    Returns:
        dict: The JSON structure ready for integration with your dataloader.
    """
    # Initialize the JSON dictionary with the required structure.
    # Each label contains separate dictionaries for train and test splits.
    dataset = {
        dataset_name: {
            "ffhq_imgs": {"train": {}, "test": {}},
            "stylegan2_psi1.0": {"train": {}, "test": {}}
        }
    }

    # Helper function to process one folder.
    def process_folder(folder, label):
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in files:
            # Create the full file path.
            full_path = os.path.join(folder, file)
            # Randomly assign to train or test.
            mode = "train" if random.random() < train_ratio else "test"
            # Use the file name (without extension) as a unique key.
            key = os.path.splitext(file)[0]
            # Each entry contains the label and a list with one frame (the image path).
            dataset[dataset_name][label][mode][key] = {
                "label": label,
                "frames": [full_path]
            }

    # Process real images folder.
    process_folder(real_folder, "ffhq_imgs")
    # Process fake images folder.
    process_folder(fake_folder, "stylegan2_psi1.0")

    return dataset

def main():
    parser = argparse.ArgumentParser(
        description="Create JSON file for new dataset from real and fake image folders."
    )
    parser.add_argument(
        "--real_folder", type=str, default="/pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/datasets/rgb/df-tp-ffhq/ffhq_imgs", #, required=True
        help="Path to folder containing real images (ffhq_imgs)."
    )
    parser.add_argument(
        "--fake_folder", type=str, default="/pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/datasets/rgb/df-tp-ffhq/stylegan2_psi1.0", #, required=True
        help="Path to folder containing fake images (stylegan2_psi1.0)."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Fraction of images assigned to train (default: 0.8)."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="df-tp-ffhq",
        help="Name of the dataset (default: df-tp-ffhq)."
    )
    parser.add_argument(
        "--output", type=str, default="/home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/preprocessing/df-tp-ffhq.json",
        help="Output file path for the JSON file (if blank, prints to stdout)."
    )
    args = parser.parse_args()

    dataset_json = create_json(
        args.real_folder, args.fake_folder, args.train_ratio, args.dataset_name
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(dataset_json, f, indent=4)
        print(f"JSON file written to {args.output}")
    else:
        # Print JSON to stdout if no output file is specified.
        print(json.dumps(dataset_json, indent=4))

if __name__ == "__main__":
    main()
