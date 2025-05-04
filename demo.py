import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
from PIL import Image, ImageDraw

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, depthdir, maskdir, calib, stride, return_depth = True):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy
    
    image_list = sorted(os.listdir(imagedir))[::stride]
    if depthdir is not None:
        depth_list = sorted(os.listdir(depthdir))[::stride]
    if maskdir is not None:
        mask_list = sorted(os.listdir(maskdir))[::stride]
    
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        
        if depthdir is not None:
            depth = np.fromfile(os.path.join(depthdir, depth_list[t]), dtype=np.uint16) / 5000.0 # required scaling factor
            depth = depth.reshape(720 // 2, 1280 // 2) # aligned depth is half resolution of rgb
            
        if maskdir is not None:
            mask = Image.open(os.path.join(maskdir, mask_list[t])) 
            mask = np.array(mask)
            mask = np.where(mask < 127, 1, 0)
            image = np.where(mask[..., None] > 0, image, 0)  # Apply mask
            
            mask_resized = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = np.where(mask_resized > 0, depth, 0)  # Apply mask
            
            
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])
            if depthdir is not None:
                depth = cv2.undistort(depth, K, calib[4:])
                


        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        
        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        
        if depthdir is not None:
            depth = cv2.resize(depth, (w1, h1))
            depth = depth[:h1-h1%8, :w1-w1%8]
            depth = np.array(depth, dtype=np.float32)
            depth = torch.as_tensor(depth)
        else:
            depth = None
        
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        if return_depth:
            yield t, image[None], intrinsics, depth 
        else:
            yield t, image[None], intrinsics


def save_reconstruction(droid, traj, reconstruction_path):

    from pathlib import Path

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    # traj_est is [translation, quaternion] for each frame
    np.save("reconstructions/{}/traj_est.npy".format(reconstruction_path), traj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--depthdir", type=str, default="auto", help="Set to 'auto' to use depth from image directory. Set to 'False' or 'None' to disable depth")
    parser.add_argument("--maskdir", type=str, default="auto", help="Set to 'auto' to use depth from image directory. Set to 'False' or 'None' to disable depth")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    args.reconstruction_path = args.imagedir.replace("/rgb", "")

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    if args.depthdir == "auto":
        args.depthdir = args.imagedir.replace("/rgb", "/depth")
    elif args.depthdir in ["False", "None", False, None]:
        args.depthdir = None
    
    if args.maskdir == "auto":
        args.maskdir = args.imagedir.replace("/rgb", "/masks")
    elif args.maskdir in ["False", "None", False, None]:
        args.maskdir = None
        

    tstamps = []
    for (t, image, intrinsics, depth) in tqdm(image_stream(args.imagedir, args.depthdir, args.maskdir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics, depth=depth)

    # Dunno why depth and mask is None here, but seems to work
    traj_est = droid.terminate(image_stream(args.imagedir, None, None, args.calib, args.stride, return_depth = False))
    
    if args.reconstruction_path is not None:
        save_reconstruction(droid, traj_est, args.reconstruction_path)

