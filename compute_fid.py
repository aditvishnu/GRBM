import argparse
import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import torch
import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image

from grbm import GRBM


def prepare_real_images(config, real_img_path):
    real_img_path = Path(real_img_path)
    real_img_path.mkdir(parents=True, exist_ok=True)

    # If already populated, skip
    if any(real_img_path.iterdir()):
        print(f"Real images already exist in {real_img_path}")
        return

    print("Preparing real dataset images...")
    dataset_name = config.get("dataset", "").lower()

    if dataset_name == "fashionmnist":
        dataset = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(config["img_mean"], config["img_std"]),
                ]
            ),
        )

    elif dataset_name == "celeba":
        dataset = datasets.CelebA(
            root="./data",
            split="test",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(config["crop_size"]),
                    transforms.Resize(config["height"]),
                    transforms.ToTensor(),
                ],
            ),
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    for idx, (img, _) in enumerate(dataset):  # type: ignore
        save_path = real_img_path / f"image_{idx:05d}.png"
        save_image(img, save_path)

        if idx % 1000 == 0:
            print(f"Saved {idx} real images")

    print(f"Finished saving {len(dataset)} real images to {real_img_path}")


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def load_model(model, ckpt_path):
    data = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(data["model"])


def initialize_model(config):
    visible_size = config["height"] * config["width"] * config["channel"]
    return GRBM(
        visible_size,
        config["hidden_size"],
        CD_step=config["CD_step"],
        CD_burnin=config["CD_burnin"],
        init_var=config["init_var"],
        inference_method=config["inference_method"],
        Langevin_step=config["Langevin_step"],
        Langevin_eta=config["Langevin_eta"],
        is_anneal_Langevin=True,
        Langevin_adjust_step=config["Langevin_adjust_step"],
    )


def compute_fid(real_path, fake_path):
    result = subprocess.run(
        ["pytorch-fid", real_path, fake_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error running pytorch-fid:", result.stderr)
        return None
    # Extract FID score from output
    for line in result.stdout.splitlines():
        if "FID:" in line:
            return float(line.strip().split("FID:")[-1])
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID from GRBM models using pytorch-fid."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="fig34/config/fashionmnist.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model-*.pt files",
    )
    parser.add_argument(
        "--real-img-path",
        type=str,
        required=True,
        help="Directory containing real images for FID",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="If specified, evaluate from this epoch",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20_000,
        help="Number of images to sample per evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    # Ensure real images exist
    prepare_real_images(config, args.real_img_path)

    results_csv = "fid_scores.csv"
    last_completed_epoch = None

    # Check if CSV already exists to resume from last epoch
    if os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
        if not df_existing.empty:
            last_completed_epoch = df_existing["epoch"].max()
            print(f"Resuming from epoch after {last_completed_epoch}...")
    else:
        # Create empty CSV with header
        pd.DataFrame(columns=["epoch", "fid"]).to_csv(results_csv, index=False)

    ckpt_paths = glob.glob(os.path.join(args.model_dir, "model-*.pt"))

    if not ckpt_paths:
        print("No checkpoints found.")
        return

    results = []
    for ckpt_path in (bar := tqdm.tqdm(sorted(ckpt_paths))):
        epoch = int(os.path.basename(ckpt_path).split("-")[1].split(".")[0])
        if epoch <= args.start_epoch:
            continue
        if last_completed_epoch is not None and epoch <= last_completed_epoch:
            continue
        model = initialize_model(config)
        model = model.cuda()
        load_model(model, ckpt_path)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with torch.no_grad():
                B, C, H, W = (
                    args.samples,
                    config["channel"],
                    config["height"],
                    config["width"],
                )
                bs = 2000
                mean = torch.tensor(config["img_mean"]).view(1, -1, 1, 1).cuda()
                std = torch.tensor(config["img_std"]).view(1, -1, 1, 1).cuda()
                for b in tqdm.trange(0, B, bs, leave=False):
                    v_init = torch.randn(bs, C, H, W).cuda()
                    v_list = model.sampling(
                        v_init, config["sampling_steps"], config["sampling_gap"]
                    )
                    images = (v_list[-1][1] * std + mean).clamp(0, 1)

                    for idx, img_tensor in enumerate(images):
                        save_path = os.path.join(tmp_dir, f"image_{b+idx:05d}.png")
                        save_image(img_tensor, save_path)
            fid = compute_fid(args.real_img_path, tmp_dir)
            if fid is not None:
                bar.set_description(f"[Epoch {epoch}] FID: {fid:.2f}")
                results.append((epoch, fid))
                pd.DataFrame([(epoch, fid)], columns=["epoch", "fid"]).to_csv(
                    results_csv, mode="a", header=False, index=False
                )

    # Save results to CSV
    if results:
        df = pd.DataFrame(results, columns=["epoch", "fid"])
        df.to_csv("fid_scores.csv", index=False)
        print("Saved FID scores to fid_scores.csv")
