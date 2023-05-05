import os
import cv2
import argparse
import subprocess

from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from canny_inference import inference

def load_controlnet_models():
    apply_canny = CannyDetector()

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(
        load_state_dict('./models/control_sd15_canny.pth', location='cuda')
    )
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return apply_canny, model, ddim_sampler


def load_prompts(prompts_path):
    with open(prompts_path, "r") as f:
        prompts = f.readlines()
    return [prompt.strip() for prompt in prompts]


def load_paths_and_labels(labels_path):
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split(",") for line in lines]


def super_res_smooth_img(input_path):
    model_arch_name = "srresnet_x4"
    model_weights_path = "./super_res_pytorch/results/pretrained_models/SRGAN_x4-ImageNet-8c4a7569.pth.tar"
    device_type = "cuda"
    output_folder = "./content/super_res_inputs/"

    os.makedirs(output_folder, exist_ok=True)

    if (
        input_path.endswith('.jpg') or 
        input_path.endswith('.png') or 
        input_path.endswith('.jpeg')
    ):
        out_basepath, _ = os.path.splitext(input_path)
        out_img_name = out_basepath.split("/")[-1] + ".jpg"
        output_path = os.path.join(output_folder, out_img_name)
        subprocess.run(['python', 
                        './super_res_pytorch/inference.py',
                        '--model_arch_name', model_arch_name,
                        '--inputs_path', input_path,
                        '--output_path', output_path,
                        '--model_weights_path', model_weights_path,
                        '--device_type', device_type])
        return output_path


def control_net_aug_img(input_path, output_path, prompt):
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_samples = 1
    image_resolution = 256
    ddim_steps = 20
    guess_mode = False
    control_strength = 0.95
    guidance_scale = 0.4
    seed = 16824
    eta = 0
    low_threshold = 20
    high_threshold = 100

    input_image = cv2.imread(input_path)
    height, width = input_image.shape[:2]

    # if the width is greater than 1000, resize the image
    if width > 1000:
        return
        print("Resizing")
        new_width = 512
        new_height = int(height * (new_width/width))
        input_image = cv2.resize(input_image, (new_width, new_height))
    
    inference(
        input_image, output_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, control_strength, guidance_scale, seed, eta, low_threshold, high_threshold, apply_canny, model, ddim_sampler
    )


def run_pipeline(input_path, output_path, prompt):
    res_path = super_res_smooth_img(input_path)
    control_net_aug_img(res_path, output_path, prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the super-resolution and control-net augmentation pipelines.')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Path to the output folder for control-net augmentation')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Path to the labels text file for control-net augmentation')
    parser.add_argument('--prompts_path', type=str, default=None,
                        help='Path to the prompts text file for control-net augmentation')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Load models to avoid reloading for each prompt
    apply_canny, model, ddim_sampler = load_controlnet_models()

    # Load labels, and prompts path
    paths_and_labels = load_paths_and_labels(args.labels_path)
    prompts = load_prompts(args.prompts_path)
    
    # Loop through images
    for path_and_label in paths_and_labels:
        for idx, prompt in enumerate(prompts):
            input_img_path, label = path_and_label
            populated_prompt = prompt.format(label)

            out_basepath, _ = os.path.splitext(input_img_path)
            out_basepath = out_basepath.split("/")[-1]
            output_img_path = args.output_folder + out_basepath + f"_{str(idx)}" + ".jpg"

            # For each image, run complete augmentation pipeline
            run_pipeline(input_img_path, output_img_path, populated_prompt)
