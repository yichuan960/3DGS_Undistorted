# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Before using you need to run
# pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install opencv-python pycocotools matplotlib onnxruntime onnx

# Example command: pyhton3 segment.py --input <input path (use correct scale images!)> --output <path to dataset> --model-type vit_b --checkpoint <path>.ckpt

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
import argparse
import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def render_masks(path, image, masks):
    decoded = [mask['segmentation'] for mask in masks]
    decoded = mask_utils.decode(decoded)
    for i, mask in enumerate(masks):
        mask['segmentation'] = decoded[:, :, i]

    plt.figure()
    plt.imshow(image)

    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m == 1] = color_mask
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def main(args: argparse.Namespace) -> None:
    print("Loading model...")

    amg_kwargs = {
        "points_per_side": None,
        "points_per_batch": None,
        "pred_iou_thresh": None,
        "stability_score_thresh": None,
        "stability_score_offset": None,
        "box_nms_thresh": None,
        "crop_n_layers": None,
        "crop_nms_thresh": None,
        "crop_overlap_ratio": None,
        "crop_n_points_downscale_factor": None,
        "min_mask_region_area": None,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle"
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    seg_path = os.path.join(args.output, 'segments')
    os.makedirs(seg_path, exist_ok=True)
    masks_path = os.path.join(seg_path, 'masks')
    rendered_path = os.path.join(seg_path, 'rendered')
    os.makedirs(masks_path, exist_ok=True)
    os.makedirs(rendered_path, exist_ok=True)

    resolution_scale = 8

    for t in tqdm(targets):
        image = cv2.imread(t)
        #resolution = [round(image.shape[1] / resolution_scale), round(image.shape[0] / resolution_scale)]
        #resolution = [431,431]
        resolution = [503, 377]
        image = cv2.resize(image, resolution)

        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        # Save masks
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(masks_path, base)
        save_file = save_base + ".json"
        with open(save_file, "w") as f:
            json.dump(masks, f)

        # Save rendering of segmentation
        render_masks(os.path.join(rendered_path, base) + '.png', image, masks)

    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
