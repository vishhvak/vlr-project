# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import torch
from torch import nn

import imgproc
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    sr_model = model.__dict__[model_arch_name](in_channels=3,
                                               out_channels=3,
                                               channels=64,
                                               num_rcb=16)
    sr_model = sr_model.to(device=device)

    return sr_model

import pywt
import numpy as np
def method1(img):
    coeffs = pywt.dwt2(img, 'haar') # apply wavelet transform
    sigma = 20 # set the standard deviation of the noise
    threshold = sigma * np.sqrt(2*np.log(len(img))) # calculate the threshold
    coeffs = list(coeffs)
    coeffs[0] = pywt.threshold(coeffs[0], threshold) # threshold the LL subband
    denoised_img = pywt.idwt2(coeffs, 'haar') # apply inverse wavelet transform
    return denoised_img

from skimage import io, filters
def main(args):
    device = choice_device(args.device_type)

    # Initialize the model
    sr_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    sr_model = load_state_dict(sr_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    sr_model.eval()

    lr_tensor = imgproc.preprocess_one_image(args.inputs_path, device)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    sr_image = cv2.medianBlur(sr_image, 5)
    
    sobelx = cv2.Sobel(sr_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(sr_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    # Normalize the Sobel output to the range [0, 255]
    sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sr_image = cv2.addWeighted(sr_image, 0.5, sobel_norm, 0.5, 0)

    cv2.imwrite(args.output_path, sr_image)

    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="srresnet_x4")
    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure/comic_lr.png",
                        help="Low-resolution image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure/comic_sr.png",
                        help="Super-resolution image path.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/SRGAN_x4-ImageNet-8c4a7569.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args)
