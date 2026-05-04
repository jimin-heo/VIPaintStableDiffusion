import argparse
import gc
import os

import numpy as np
import torch
from PIL import Image

from sd3_infer import load_model
from VIPaint import VIPaintSampler


# Experiment presets. ``mask_path`` may be either a single file (used for all images)
# or a directory of per-image masks named ``mask_{i:06d}.png``. ``mask_is_dir`` flags
# which case we are in. All paths are relative to this file's directory.
EXPERIMENTS = {
    "synthetic": {
        "image_dir": "dataset/synthetic_data",
        "mask_path": "dataset/rectangular_mask_1024.png",
        "prompt_file": "dataset/synthetic_prompt.txt",
        "mask_is_dir": False,
    },
    "synthetic_random_mask": {
        "image_dir": "dataset/synthetic_data",
        "mask_path": "dataset/random_masks",
        "prompt_file": "dataset/synthetic_prompt.txt",
        "mask_is_dir": True,
    },
    "synthetic_simple": {
        "image_dir": "dataset/synthetic_simple_data",
        "mask_path": "dataset/rectangular_mask_1024.png",
        "prompt_file": "dataset/synthetic_simple_prompt.txt",
        "mask_is_dir": False,
    },
}


def main():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Current working directory:", os.getcwd())
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of an angry cat with an astronaut suit")
    # parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--learning_rate", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--l1_weight", type=float, default=0.05)
    parser.add_argument("--kl_weight", type=float, default=0.5)
    # parser.add_argument("--klT_weight", type=float, default=0.5)
    parser.add_argument("--lpips_weight", type=float, default=0.0)
    parser.add_argument("--mid_weight", type=float, default=0.5)
    parser.add_argument("--rec_weight", type=float, default=6.0)
    parser.add_argument("--bounds", nargs=2, type=int, default=[250, 500])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--dps_scale", type=float, default=650.0)
    parser.add_argument("--outdir", type=str, default="results/vipaint_output")
    parser.add_argument("--experiment", type=str, default="synthetic", choices=list(EXPERIMENTS.keys()), help="Predefined experiment that selects image_dir, mask_path, and prompt_file.")
    parser.add_argument("--image_dir", type=str, default=None, help="Override the experiment's image directory.")
    parser.add_argument("--mask_path", type=str, default=None, help="Override the experiment's mask path (single file or directory).")
    parser.add_argument("--prompt_file", type=str, default=None, help="Override the experiment's prompt file.")
    parser.add_argument("--version", type=str, default="sd3.5_large_turbo", help="SD3 model version.")
    args = parser.parse_args()

    cfg = {
        "prompt": args.prompt,
        # "cfg_scale": 4.5,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "K": args.K,
        "l1_weight": args.l1_weight,
        "kl_weight": args.kl_weight,
        # "klT_weight": args.klT_weight,
        "lpips_weight": args.lpips_weight,
        "mid_weight": args.mid_weight,
        "rec_weight": args.rec_weight,
        "bounds": args.bounds,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "dps_scale": args.dps_scale,
    }

    print("[INFO] Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print(f"[INFO] Output dir:  {args.outdir}")
    print(f"[INFO] Experiment:  {args.experiment}")

    directory = os.path.join(base_dir, args.outdir)
    os.makedirs(directory, exist_ok=True)

    # Resolve data paths from the chosen experiment, with optional CLI overrides.
    exp = EXPERIMENTS[args.experiment]
    image_dir = os.path.join(base_dir, args.image_dir or exp["image_dir"])
    mask_path_or_dir = os.path.join(base_dir, args.mask_path or exp["mask_path"])
    prompt_file = os.path.join(base_dir, args.prompt_file or exp["prompt_file"])
    mask_is_dir = exp["mask_is_dir"] if args.mask_path is None else os.path.isdir(mask_path_or_dir)

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines()]

    inferencer = load_model(args.version)

    files = sorted(os.listdir(image_dir))
    num_samples = min(len(files), len(prompts))

    for i in range(2):
        print(f"[INFO] Processing image {i}")
        output_root = os.path.join(base_dir, args.outdir, f"{i:02d}")
        os.makedirs(output_root, exist_ok=True)

        image = files[i]

        image_path = os.path.join(image_dir, image)
        if mask_is_dir:
            mask_path = os.path.join(mask_path_or_dir, f"mask_{i:06d}.png")
        else:
            mask_path = mask_path_or_dir
        prompt = prompts[i]
        cfg["prompt"] = prompt

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_np = np.array(img)
        mask_np = np.array(mask)

        # mask_np = (mask_np[:, :, None] // 255)
        # masked = img_np * mask_np

        # Fill the masked region with mean color + Gaussian noise.
        mask_np = (mask_np > 127).astype(np.uint8)[:, :, None]
        # mask_np = 1 - mask_np
        avg_color_rgb = np.mean(img_np, axis=(0, 1)).astype(np.uint8)
        sigma = 40.0
        noise = np.random.normal(0, sigma, size=img_np.shape)
        fill = np.clip(avg_color_rgb + noise, 0, 255).astype(np.uint8)
        masked = np.where(mask_np.astype(bool), img_np, fill)

        image_save_path = os.path.join(output_root, f"{i:03d}_img.png")
        mask_save_path = os.path.join(output_root, f"{i:03d}_masked.png")
        prompt_save_path = os.path.join(output_root, f"{i:03d}_prompt.txt")

        Image.fromarray(masked).save(mask_save_path)
        Image.fromarray(img_np).save(image_save_path)
        with open(prompt_save_path, "w", encoding="utf-8") as f:
            f.write(prompt + "\n\n")
            for key, value in cfg.items():
                f.write(f"{key}: {value}\n")

        sampler = VIPaintSampler(masked, mask_np, inferencer, cfg)
        sampler.optimize(i, directory, output_root)
        sampler.sample(i, directory, output_root)

        del sampler
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
