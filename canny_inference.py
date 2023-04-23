
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torchc
import random
import argparse

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))

# Uncomment if using your own weights
# weights_path = None
# print("Loading Checkpoint Weights ...")
# model.load_state_dict(torch.load(weights_path)
# print("Loading Checkpoint Weights Successful!!!")

model = model.cuda()
ddim_sampler = DDIMSampler(model)

def inference(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    #return [255 - detected_map] + results

    # Save the resulting images to the output path
    for i in range(len(results)):
        cv2.imwrite(args.output_path + str(i) + '.jpg', results[i])

# Write main function below with if __name__ == '__main__'
if __name__ == '__main__':
    # Get image_path from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='./images/')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--a_prompt', type=str, default='') 
    parser.add_argument('--n_prompt', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--image_resolution', type=int, default=256)
    parser.add_argument('--ddim_steps', type=int, default=20)
    parser.add_argument('--guess_mode', type=bool, default=False)
    parser.add_argument('--control_strength', type=float, default=0.5)
    parser.add_argument('--guidance_scale', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=16824)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--low_threshold', type=float, default=100)
    parser.add_argument('--high_threshold', type=float, default=200)
    parser.add_argument('--output_folder', type=str, default='./outputs/')
    args = parser.parse_args()

    # Get all jpg files in the image_folder using glob
    image_files = os.listdir(args.image_folder)
    image_paths = [os.path.join(args.image_folder, image_file) for image_file in image_files]

    # Loop through all the image paths
    for image_path in image_paths:
        # Read the image using cv2.imread
        input_image = cv2.imread(image_path)
        # Call inference function
        inference(input_image, args.prompt, args.a_prompt, args.n_prompt, args.num_samples, args.image_resolution, args.ddim_steps, args.guess_mode, args.control_strength, args.guidanec_scale, args.seed, args.eta, args.low_threshold, args.high_threshold)
    

    
