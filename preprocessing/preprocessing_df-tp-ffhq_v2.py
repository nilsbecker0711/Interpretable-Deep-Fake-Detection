import os
import json
import argparse

def create_json(real_folder, fake_folder, dataset_name="df-tp-ffhq"):
    """
    Creates a JSON structure for the new dataset with a 70/20/10 train/val/test split.

    Args:
        real_folder (str): Path to the folder with real images (ffhq_imgs).
        fake_folder (str): Path to the folder with fake images (stylegan2_psi1.0).
        dataset_name (str): Name for the dataset (default "df-tp-ffhq").

    Returns:
        dict: The JSON structure ready for integration with your dataloader.
    """
    # Initialize the JSON dictionary structure with train, val, and test keys.
    dataset = {
        dataset_name: {
            "ffhq_imgs": {"train": {}, "val": {}, "test": {}},
            "stylegan2_psi1.0": {"train": {}, "val": {}, "test": {}}
        }
    }

    def process_folder(folder, label):
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()  # ensure consistent split order
        total = len(files)
        n_train = int(total * 0.7)
        n_val = int(total * 0.2)
        n_test = total - n_train - n_val  # remainder to test

        split_map = (
            ("train", files[:n_train]),
            ("val", files[n_train:n_train + n_val]),
            ("test", files[n_train + n_val:])
        )

        for mode, split_files in split_map:
            for file in split_files:
                full_path = os.path.join(folder, file)
                key = os.path.splitext(file)[0]
                dataset[dataset_name][label][mode][key] = {
                    "label": label,
                    "frames": [full_path]
                }

    process_folder(real_folder, "ffhq_imgs")
    process_folder(fake_folder, "stylegan2_psi1.0")

    return dataset

def main():
    parser = argparse.ArgumentParser(
        description="Create JSON file for dataset with 70/20/10 split."
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
        "--dataset_name", type=str, default="df-tp-ffhq",
        help="Name of the dataset (default: df-tp-ffhq)."
    )
    parser.add_argument(
        "--output", type=str, default="/home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/preprocessing/df-tp-ffhq.json",
        help="Output file path for the JSON file."
    )
    args = parser.parse_args()

    dataset_json = create_json(
        args.real_folder, args.fake_folder, args.dataset_name
    )

    with open(args.output, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"JSON file written to {args.output}")

if __name__ == "__main__":
    main()
