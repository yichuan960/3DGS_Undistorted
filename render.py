#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json

import torch
from scene import Scene
import os
from utils import image_utils
from utils import loss_utils
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    # Losses & Metrics.
    # ssim_f = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    # psnr_f = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    lpips_f = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        pixels = gt.unsqueeze(0)
        colors = rendering[:3, ...].unsqueeze(0)
        pixels = torch.clamp(pixels, 0.0, 1.0)
        colors = torch.clamp(colors, 0.0, 1.0)
        metrics["psnr"].append(image_utils.psnr(rendering, gt).mean())
        metrics["ssim"].append(loss_utils.ssim(rendering, gt).mean())
        metrics["lpips"].append(lpips_f(colors, pixels))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    psnr = torch.tensor(metrics["psnr"]).mean()
    ssim = torch.tensor(metrics["ssim"]).mean()
    lpips = torch.stack(metrics["lpips"]).mean()
    # save stats as json
    stats = {
        "psnr": psnr.item(),
        "ssim": ssim.item(),
        "lpips": lpips.item(),
    }
    if name == "train":
        with open(f"{model_path}/val_step_train_30000.json", "w") as f:
            json.dump(stats, f)
    if name == "test":
        with open(f"{model_path}/val_step_test_30000.json", "w") as f:
            json.dump(stats, f)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                config: dict):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, config, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config", type=str, default="config.json")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Load robust Gaussians config
    with open(args.config, 'r') as file:
        robust_params = json.load(file)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                robust_params)
