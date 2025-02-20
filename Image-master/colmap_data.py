#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath
import transform_colmap_camera

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="convert transforms.json to colmap.txt")
    parser.add_argument("--camera_module", default="PINHOLE", help="camera module type")
    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument("--colmap_db", default="database.db", help="colmap database filename")
    args = parser.parse_args()
    return args

def create_output():
    # Setup output directories.
    output_dir = f"/sparse/0/"
    os.makedirs(output_dir, exist_ok=True)
    camera_txt = f"/sparse/0/cameras.txt"
    image_txt = f"/sparse/0/images.txt"
    points3D_txt = f"/sparse/0/points3D.txt"
    file1 = open(camera_txt, "w")
    file1.close()
    file2 = open(image_txt, "w")
    file2.close()
    file3 = open(points3D_txt, "w")
    file3.close()
    
def do_system(arg):
    print(f"==== running: {arg}")
    err=os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def go_camera2colmap(args):
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    images_path = args.images + "/images"
    fnames = list(sorted(os.listdir(images_path)))
    fname2pose = {}
    transform_path = args.images + "/transforms.json"
    with open(transform_path, 'r') as f:
        meta = json.load(f)
    
    H = meta['h']
    W = meta['w']
    k1, k2, k3, k4, p1, p2 = meta['k1'], meta['k2'], meta['k3'], meta['k4'], meta['p1'], meta['p2']
    fx = 0.5 * W / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
    if 'camera_angle_y' in meta:
        fy = 0.5 * H / np.tan(0.5 * meta['camera_angle_y'])  # original focal length
    else:
        fy = fx
    if 'cx' in meta:
        cx, cy = meta['cx'], meta['cy']
    else:
        cx = 0.5 * W
        cy = 0.5 * H
    camera_path = args.images + "/sparse/0/cameras.txt"
    with open(camera_path, 'w') as f:
        if args.camera_module == "PINHOLE":
            f.write(f'1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
        elif args.camera_module == "SIMPLE_PINHOLE":
            f.write(f'1 SIMPLE_PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
        elif args.camera_module == "SIMPLE_RADIAL":
            f.write(f'1 SIMPLE_PINHOLE {W} {H} {fx} {fy} {cx} {cy} {k1}')
        elif args.camera_module == "RADIAL":
            f.write(f'1 SIMPLE_PINHOLE {W} {H} {fx} {fy} {cx} {cy} {k1} {k2}')
        elif args.camera_module == "OPENCV":
            f.write(f'1 OPENCV {W} {H} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}')
        elif args.camera_module == "OPENCV_FISHEYE":
            f.write(f'1 OPENCV {W} {H} {fx} {fy} {cx} {cy} {k1} {k2} {k3} {k4}')
        idx = 1
        for frame in meta['frames']:
            fname = frame['file_path'].split('/')[-1]
            if not (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.JPG')):
                fname += '.png'
            # blend to opencv
            pose = np.array(frame['transform_matrix']) @ blender2opencv
            fname2pose.update({fname: pose})

    image_path = args.images + "/sparse/0/images.txt"
    with open(image_path, 'w') as f:
        for fname in fnames:
            pose = fname2pose[fname]
            # blender：world = R * camera + T; colmap：camera = R * world + T
            # R’ = R^-1
            # t’ = -R^-1 * t
            R = np.linalg.inv(pose[:3, :3])
            T = -np.matmul(R, pose[:3, 3])
            q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    
            f.write(f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n')
            idx += 1

    points_path = args.images + "/sparse/0/points3D.txt"
    with open(points_path, 'w') as f:
       f.write('')

def extract_features(args):
    db = args.colmap_db
    images = args.images + "/images"
    do_system(f"colmap feature_extractor --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --database_path {db} --image_path {images}")

def run_colmap(args):
    db=args.colmap_db
    images = args.images + "/images"
    input_path = args.images + "/sparse/0"
    output_path = args.images + "/sparse/0"
    do_system(f"colmap exhaustive_matcher --database_path {db}")
    do_system(f"colmap point_triangulator --database_path {db} --image_path {images} --input_path {input_path} --output_path {output_path}")

def transform_data():
    args = parse_args()
    # Output /sparse/0
    create_output()
    # Get camera Intrinsics & Extrinsics
    go_camera2colmap(args)
    # extract features
    extract_features(args)
    print(f"extract_features finish")
    transform_colmap_camera.camTodatabase("sparse/0/cameras.txt")
    # match features
    run_colmap(args)
    print(f"outputting to sparse/0")
	
if __name__ == "__main__":
    transform_data()