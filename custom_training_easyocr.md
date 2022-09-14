### 1. Training the Recognition Model

There are 2 options to train your own recognition model:

**1. Open-source approach**

For the open-source approach, you can use your own data or generate your own dataset. To generate your own data, we recommend using
[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). We provide an example of a dataset [here](https://jaided.ai/easyocr/modelhub/).
After you have a dataset, you can train your own model by following this repository
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
The network needs to be fully convolutional in order to predict flexible text length. Our current network is 'None-VGG-BiLSTM-CTC'.
Once you have your trained model (a `.pth` file), you need 2 additional files describing recognition network architecture and model configuration.
An example is provided in `custom_example.zip` file [here](https://jaided.ai/easyocr/modelhub/).

Please do not create an issue about data generation and model training in this repository. If you have any question regarding data generation and model training, please ask in the respective repositories.

Note: We also provide our version of a training script [here](https://github.com/JaidedAI/EasyOCR/tree/master/trainer). It is a modified version from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

**2. Web-based approach**

Jaided AI provides a web-based (paid) service for training your own model [here](https://jaided.ai/). You can train your model on the cloud and export it for local deployment. All 3 files are downloadable once the model has finished training.

## How to use your custom model

To use your own recognition model, you need the three files from the open-source or web-based approach above. These three files have to share the same name (i.e. `yourmodel.pth`, `yourmodel.yaml`, `yourmodel.py`) that you will then use to call your model with in the EasyOCR API.

We provide [custom_example.zip](https://jaided.ai/easyocr/modelhub/)
as an example. Please download, extract and place `custom_example.py`, `custom_example.yaml` in the `user_network_directory` (default = `~/.EasyOCR/user_network`) and place `custom_example.pth` in model directory (default = `~/.EasyOCR/model`)
Once you place all 3 files in their respective places, you can use `custom_example` by
specifying `recog_network` like this `reader = easyocr.Reader(['en'], recog_network='custom_example')`.


### 2. Training the Detection Model (Training the CRAFT)
On the official CRAFT github, there are many people who want to train CRAFT models. 

However, the training code is not published in the official CRAFT repository. 

There are other reproduced codes, but there is a gap between their performance and performance reported in the original paper. (https://arxiv.org/pdf/1904.01941.pdf) 

The trained model with this code recorded a level of performance similar to that of the original paper.

```bash
├── config
│   ├── syn_train.yaml
│   └── custom_data_train.yaml
├── data
│   ├── pseudo_label
│   │   ├── make_charbox.py
│   │   └── watershed.py
│   ├── boxEnlarge.py
│   ├── dataset.py
│   ├── gaussian.py
│   ├── imgaug.py
│   └── imgproc.py
├── loss
│   └── mseloss.py
├── metrics
│   └── eval_det_iou.py
├── model
│   ├── craft.py
│   └── vgg16_bn.py
├── utils
│   ├── craft_utils.py
│   ├── inference_boxes.py
│   └── utils.py
├── trainSynth.py
├── train.py
├── train_distributed.py
├── eval.py
├── data_root_dir   (place dataset folder here)
└── exp             (model and experiment result files will saved here)
```

### Installation

Install using `pip`

``` bash
pip install -r requirements.txt
```


### Training
1. Put your training, test data in the following format
    ```
    └── data_root_dir (you can change root dir in yaml file)
        ├── ch4_training_images
        │   ├── img_1.jpg
        │   └── img_2.jpg
        ├── ch4_training_localization_transcription_gt
        │   ├── gt_img_1.txt
        │   └── gt_img_2.txt
        ├── ch4_test_images
        │   ├── img_1.jpg
        │   └── img_2.jpg
        └── ch4_training_localization_transcription_gt
            ├── gt_img_1.txt
            └── gt_img_2.txt
    ```
   * localization_transcription_gt files format :
   ```
    377,117,463,117,465,130,378,130,Genaxis Theatre
    493,115,519,115,519,131,493,131,[06]
    374,155,409,155,409,170,374,170,###
    ```
2. Write configuration in yaml format (example config files are provided in `config` folder.)
    * To speed up training time with multi-gpu, set num_worker > 0   
3. Put the yaml file in the config folder
4. Run training script like below (If you have multi-gpu, run train_distributed.py)
5. Then, experiment results will be saved to ```./exp/[yaml]``` by default.

* Step 1 : To train CRAFT with SynthText dataset from scratch
    * Note : This step is not necessary if you use <a href="https://drive.google.com/file/d/1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ/view?usp=sharing">this pretrain</a> as a checkpoint when start training step 2. You can download and put it in `exp/CRAFT_clr_amp_29500.pth` and change `ckpt_path` in the config file according to your local setup.
    ```
    CUDA_VISIBLE_DEVICES=0 python3 trainSynth.py --yaml=syn_train
    ```

* Step 2 : To train CRAFT with [SynthText + IC15] or custom dataset
    ```
    CUDA_VISIBLE_DEVICES=0 python3 train.py --yaml=custom_data_train               ## if you run on single GPU
    CUDA_VISIBLE_DEVICES=0,1 python3 train_distributed.py --yaml=custom_data_train   ## if you run on multi GPU
    ```

### Arguments
* ```--yaml``` : configuration file name

### Evaluation
* In the official repository issues, the author mentioned that the first row setting F1-score is around 0.75.
* In the official paper, it is stated that the result F1-score of the second row setting is 0.87.
    * If you adjust post-process parameter 'text_threshold' from 0.85 to 0.75, then F1-score reaches to 0.856.
* It took 14h to train weak-supervision 25k iteration with 8 RTX 3090 Ti.
    * Half of GPU assigned for training, and half of GPU assigned for supervision setting.

| Training Dataset   | Evaluation Dataset   | Precision  | Recall  | F1-score  | pretrained model  |
| ------------- |-----|:-----:|:-----:|:-----:|-----:|
| SynthText      |  ICDAR2013 | 0.801 | 0.748 | 0.773| <a href="https://drive.google.com/file/d/1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ/view?usp=sharing">download link</a>|
| SynthText + ICDAR2015      | ICDAR2015  | 0.909 | 0.794 | 0.848| <a href="https://drive.google.com/file/d/1qUeZIDSFCOuGS9yo8o0fi-zYHLEW6lBP/view">download link</a>|