

## Installation Setup

Download models and annotator ckpts from here
- `wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth`, extract the file into the models folder
- `curl -L https://cmu.box.com/shared/static/02o1y4bu2a4r2x5nrfuxx5v0zaa1l2wr --output annotator_ckpts.tar`, move the contents of the extracted folder into the ckpts folder under the annotator folder.


## To Run Control-Net Data Augmentation Pipeline
- mkdir "content/"
- unzip image.zip into "content/"
- After unzipping your images are in "content/MJ_train/"
- Create a labels file of the format:
```
./content/MJ_train_5000_Adithya/image-000000002.jpg,spencerian
./content/MJ_train_5000_Adithya/image-000000003.jpg,accommodatingly
```
- Create a prompts.txt file, here's a complete sample:
```
'The word "{}" with filled red alphabets on plain background'
```
- run `python pipeline.py` by supplying paths to --output_folder, --labels_path and --prompts_path 


##To run Experiments
- Clone repository https://github.com/roatienza/deep-text-recognition-benchmark
- Replace dataset.py with the new dataset.py from this repository
- Run the train command 

```
RANDOM=$$

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training \
--valid_data data_lmdb_release/evaluation --select_data MJ-ST \
--batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None \ 
--SequenceModeling None --Prediction None --Transformer \
--TransformerModel=vitstr_tiny_patch16_224 --imgH 224 --imgW 224 \
--manualSeed=$RANDOM  --sensitive
```
