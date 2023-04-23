

## Installation Setup

Download models and annotator ckpts from here
- `wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth`, extract the file into the models folder
- `curl -L https://cmu.box.com/shared/static/02o1y4bu2a4r2x5nrfuxx5v0zaa1l2wr --output annotator_ckpts.tar`, move the contents of the extracted folder into the ckpts folder under the annotator folder.


## To Run Control-Net Data Augmentation Pipeline
- Create a "data/input" folder and place images to be augmented there
- Inside the "data/" folder place input_labels.txt with the following format
```
./data/super_res_inputs/a.jpg,culver
./data/super_res_inputs/b.jpg,BOUNDARIES
./data/super_res_inputs/c.jpg,vaunt
./data/super_res_inputs/d.jpg,INSULATED
```
- run `python pipeline.py`
