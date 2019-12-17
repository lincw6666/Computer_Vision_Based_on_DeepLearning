DLCV Homework 04
===

# Overview

- Directory tree
- How to run
- File description

----

# Directory tree

![](https://i.imgur.com/rXZu1rm.png)

----

# How to run

- Clone the repository from my github (or download `HW04.zip` then extract it).
    ```shell=
    git clone https://github.com/lincw6666/Computer_Vision_Based_on_DeepLearning.git --branch HW04 --single-branch --depth 1
    ```
- Setup the environment.
    ```shell=
    sh setup.py
    ```
    **Remark**: Execute the command under directory `HW04`.
- Start training.
    ```shell=
    python train.py
    ```
    It'll create a folder `train_logs`. All the training info will store at here.
- Predict on test data.
    - `--model_pth`: The path to your model weight. Default `"./model_weight/model.pth"`
    - Command
        ```shell=
        python test.py --model_pth path_to_your_weight
        ```
        or
        ```shell=
        python test.py
        ```
    - Generate `output.json` under `train_logs/`.

----

# Files description

- **configs/**: Store all the config files.
- **data/**
    - Including the annotation files and train/test dataset.
    - After running *setup.sh*, the zip files will be extracted.
- **model_weight/**: The pre-trained model of HW04.
- **results/**: The submission files of HW04.
- **Mask_RCNN.ipynb**: Jupyter notebook for development on Google CoLab. So there're some blocks such as mounting google drive, which is useless when you run locally.
- **setup.sh**: Build the environment.
- **train.py**: Train the model.
- **test.py**: Predict on the test data.
- **utils.py**: Contain function which transform the binary mask to RLE format.
