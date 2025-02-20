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

import os

import imageio
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from robust_loss import calculate_mask_proba, RobustLoss
from robust_loss import calculate_mask as calculate_mask_old
from segment_overlap import segment_overlap
from torchvision.transforms import ToPILImage
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
import json
import gc

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def robust_mask(
    error_per_pixel: torch.Tensor, loss_threshold: float
) -> torch.Tensor:
    epsilon = 1e-3
    error_per_pixel = error_per_pixel.mean(axis=-1, keepdims=True) #[1,431,431,1]
    error_per_pixel = error_per_pixel.squeeze(-1).unsqueeze(0) # [1,1,431,431]
    is_inlier_pixel = (error_per_pixel < loss_threshold).float()
    window_size = 3
    channel = 1
    window = torch.ones((1, 1, window_size, window_size), dtype=torch.float) / (
        window_size * window_size
    )
    if error_per_pixel.is_cuda:
        window = window.cuda(error_per_pixel.get_device())
    window = window.type_as(error_per_pixel)
    has_inlier_neighbors = F.conv2d(
        is_inlier_pixel, window, padding=window_size // 2, groups=channel
    ) # [1,1,431,431]
    has_inlier_neighbors = (has_inlier_neighbors > 0.5).float()
    is_inlier_pixel = ((has_inlier_neighbors + is_inlier_pixel) > epsilon).float()
    pred_mask = is_inlier_pixel.squeeze(0).unsqueeze(-1) # [1,431,431,1]
    return pred_mask

def robust_cluster_mask(inlier_sf, semantics):
    inlier_sf = inlier_sf.squeeze(-1).unsqueeze(0)
    cluster_size = torch.sum(semantics, axis=[-1, -2], keepdims=True, dtype=torch.float)
    inlier_cluster_size = torch.sum(inlier_sf * semantics, axis=[-1, -2], keepdims=True, dtype=torch.float)
    cluster_inlier_percentage = (inlier_cluster_size / cluster_size).float()
    is_inlier_cluster = (cluster_inlier_percentage > 0.5).float()
    inlier_sf = torch.sum(semantics * is_inlier_cluster, axis=1, keepdims=True, dtype=torch.float)
    pred_mask = inlier_sf.squeeze(0).unsqueeze(-1)
    return pred_mask


def training(dataset, opt, pipe, config, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from):
    torch.cuda.empty_cache()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, dataset, opt, pipe, config)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, config)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    epoch = 0
    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    running_stats = {
        "hist_err": torch.zeros((10000,)),
        "avg_err": 1.0,
    }

    uid_to_image_name = np.empty(len(viewpoint_stack), dtype=object)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            epoch += 1
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        image_id = viewpoint_cam.uid

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand(3, device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        pixels = gt_image

        # Spotless colors
        colors = image[:3, ...]

        first_colors = colors.clone().permute(1,2,0).unsqueeze(0)

        # if viewpoint_cam.image_name == "2clutter1":
        #     print(pixels.permute(1,2,0).unsqueeze(0))

        error_per_pixel = torch.abs(colors - pixels) # colors pixels error_per_pixel [3,431,431]
        error_per_pixel = error_per_pixel.permute(1,2,0).unsqueeze(0) # [1,431,431,3]
        log_error_per_pixel = error_per_pixel.clone()
        # print(error_per_pixel)
        pred_mask = robust_mask(
            error_per_pixel, running_stats["avg_err"]
        )
        log_pred_mask_test = pred_mask.clone()
        semantics = torch.from_numpy(np.array(viewpoint_cam.features[0])).float() # 100 50 50
        sf = semantics.to("cuda")
        sf_t = sf.unsqueeze(0)
        # cluster the semantic feature and mask based on cluster voting
        sf_f = nn.Upsample(
            size=(colors.shape[1], colors.shape[2]),
            mode="nearest",
        )(sf_t).squeeze(0) #[100 431 431]
        pred_mask = robust_cluster_mask(pred_mask, semantics=sf_f)
        input_mask = pred_mask.clone()
        log_pred_mask = pred_mask.clone()


        #if config["use_segmentation"]:
        mask_s = segment_overlap(input_mask.squeeze(), viewpoint_cam.segments, config).to('cuda') #[432,432]
        log_mask_s = mask_s.clone().unsqueeze(-1).unsqueeze(0)

        # combine spotless mask with robust mask
        mask = (mask_s.clone().unsqueeze(-1).unsqueeze(0)) * pred_mask #[1,378,504,1]
        log_mask = mask.clone()

        rgbloss = (mask.clone().detach() * error_per_pixel).mean()
        loss = rgbloss
        loss.backward()

        uid_to_image_name[viewpoint_cam.uid] = viewpoint_cam.image_name
        iter_end.record()
        running_stats["err"] = torch.histogram(
                torch.mean(torch.abs(colors.clone().permute(1,2,0).unsqueeze(0) - pixels.clone().permute(1,2,0).unsqueeze(0)), dim=-3).clone().detach().cpu(),
                bins=10000,
                range=(0.0, 1.0),
            )[0]
        new_his,new_avg = update_running_stats(running_stats)
        running_stats["hist_err"] = new_his
        running_stats["avg_err"] = new_avg
        if iteration % config["save_mask_interval"] == 0 or iteration == 1:
            path = os.path.join(scene.model_path, 'masks')
            #log_mask_path = os.path.join(path, 'log_mask')
            seg_mask_path = os.path.join(path, 'seg_mask')
            #before_log_mask_path = os.path.join(path, 'before_log')
            pre_mask_path = os.path.join(path, 'pre_log')

            if not os.path.exists(os.path.join(scene.model_path, 'masks')):
                os.mkdir(path)
                #os.mkdir(log_mask_path)
                os.mkdir(seg_mask_path)
                #os.mkdir(before_log_mask_path)
                os.mkdir(pre_mask_path)

            rgb_pred_mask_1 = (
                (log_pred_mask_test > 0.5).repeat(1, 1, 1, 3).clone().detach()
            )

            rgb_pred_mask_2 = (
                (log_pred_mask > 0.5).repeat(1, 1, 1, 3).clone().detach()
            )
            rgb_mask_s = (
                (log_mask_s > 0.5).repeat(1, 1, 1, 3).clone().detach()
            )
            rgb_mask = (
                (log_mask > 0.5).repeat(1, 1, 1, 3).clone().detach()
            )
            log_pixels = pixels.clone().permute(1,2,0).unsqueeze(0)
            log_colors = colors.clone().permute(1,2,0).unsqueeze(0)
            if iteration == 1:
                temp = torch.cat([log_pixels, first_colors, log_error_per_pixel, rgb_pred_mask_1, rgb_pred_mask_2, rgb_mask_s, rgb_mask, log_colors], dim=2)
            else:
                temp = torch.cat(
                    [log_pixels, log_error_per_pixel, rgb_pred_mask_1, rgb_pred_mask_2, rgb_mask_s,
                     rgb_mask, log_colors], dim=2)
            canvas = (
                temp
                .squeeze(0)
                .cpu()
                .detach()
                .numpy()
            )
            imageio.imwrite(
                f"{pre_mask_path}/train_{viewpoint_cam.image_name}.png",
                (canvas * 255).astype(np.uint8),
            )

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
            #                testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if not os.path.exists(os.path.join(scene.model_path, 'masks')):
        path = os.path.join(scene.model_path, 'masks')
        os.mkdir(path)

def update_running_stats(info):
    """Update running stats."""
    info["hist_err"] = (
        0.95 * info["hist_err"] + info["err"]
    )
    mid_err = torch.sum(info["hist_err"]) * 0.7
    info["avg_err"] = torch.linspace(0, 1, 10000 + 1)[
        torch.where(torch.cumsum(info["hist_err"], 0) >= mid_err)[0][
            0
        ]
    ]
    return info["hist_err"],info["avg_err"]

def prepare_output_and_logger(args, dataset, opt, pipe, config):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save training configuration
    print(f"Saving training config at {os.path.join(args.model_path, 'config.json')}")
    config = {"Model Params": dataset.__dict__, "Optimization Params": opt.__dict__, "Pipeline Params": pipe.__dict__,
              "Robust Params": config}
    with open(os.path.join(args.model_path, "config.json"), 'w') as file:
        json.dump(config, file, indent=4, sort_keys=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load robust Gaussians config
    with open(args.config, 'r') as file:
        robust_params = json.load(file)
    if not 'debug' in robust_params:
        robust_params['debug'] = False

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), robust_params, args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
