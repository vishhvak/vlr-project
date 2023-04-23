import os
import cv2
import shutil
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

    # Uncomment if using your own weights
    # weights_path = None
    # print("Loading Checkpoint Weights ...")
    # model.load_state_dict(torch.load(weights_path)
    # print("Loading Checkpoint Weights Successful!!!")

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return apply_canny, model, ddim_sampler


def super_res_smooth():
    # Set the arguments for the super resolution script
    model_arch_name = "srresnet_x4"
    model_weights_path = "./super_res_pytorch/results/pretrained_models/SRGAN_x4-ImageNet-8c4a7569.pth.tar"
    device_type = "cuda"
    input_folder = "./data/inputs/"
    output_folder = "./data/super_res_inputs/"

    assert os.path.exists(input_folder), "No input data provided!"
    os.makedirs(output_folder, exist_ok=True)

    # Loop over all the images in the input folder and call the super resolution script on each one
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            subprocess.run(['python', 
                            './super_res_pytorch/inference.py',
                            '--model_arch_name', model_arch_name,
                            '--inputs_path', input_path,
                            '--output_path', output_path,
                            '--model_weights_path', model_weights_path,
                            '--device_type', device_type])


def load_prompts():
    with open("prompts.txt", "r") as f:
        prompts = f.readlines()
    return [prompt.strip() for prompt in prompts]


def load_paths_and_labels():
    with open('./data/input_labels.txt', 'r') as f:
        lines = f.readlines()
    return [line.strip().split(",") for line in lines]


def control_net_aug():
    # Load models to avoid reloading for each prompt
    apply_canny, model, ddim_sampler = load_controlnet_models()

    image_folder = "./data/super_res_inputs/"
    output_folder = "./data/final_output/"
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_samples = 1
    image_resolution = 256
    ddim_steps = 20
    guess_mode = False
    control_strength = 0.8
    guidance_scale = 0.8
    seed = 16824
    eta = 0
    low_threshold = 40
    high_threshold = 80

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    prompts = load_prompts()
    paths_and_labels = load_paths_and_labels()
    for path_and_label in paths_and_labels:
        for idx, prompt in enumerate(prompts):
            image_path, label = path_and_label
            print(image_path, label)
            populated_prompt = prompt.format(label)
            print(f"Current prompt: {populated_prompt}")
            input_image = cv2.imread(image_path)
            out_basepath, _ = os.path.splitext(image_path)
            out_basepath = out_basepath.split("/")[-1]
            output_image = output_folder + out_basepath + str(idx) + ".jpg"
            
            inference(
                input_image, output_image, populated_prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, control_strength, guidance_scale, seed, eta, low_threshold, high_threshold, apply_canny, model, ddim_sampler
            )
    shutil.rmtree("./data/super_res_inputs")


if __name__ == "__main__":
    # Run the super-resolution + smoothing pipeline
    super_res_smooth()

    # Run control-net augmentation
    control_net_aug()
