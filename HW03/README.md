DLCV Homework 03
===

# Files description

- **DLCV_HW03_RetinaNet.ipynb**: This jupyter notebook is what I'd done on the CoLab. So there're some blocks such as mounting google drive, which is useless when you run locally. This file is for development.
- **src**
    - **fcos_train.py**: It trains a FCOS on your dataset.
    - **evaluation.py**: It detects the number in the given single image.
    - **dump_result_to_json.py**: It stores the results of detecting all the images in the SVHN testing dataset.
    - **get_annote.py**: It parses the bounding boxes information in the `.mat` annotation file. Then, it stores the parsed information into `.pkl` file.
    - **show_bbox.py**: It shows the annotation on the image. The purpose is to check that whether we parse the annotation file correctly.
- **config**
    - **retinanet_r101_fpn_1x.py**: The configuration file for RetinaNet. 'r101' stands for using ResNet101.
    - **fcos_mstrain_640_288_resnet50_fpn_gn_2.py**: The configuration file for FCOS.
- **mmdetection_patch**
    - **loading.py**: There's a bug in `mmdetection`. When we load the annotation file with no `bboxx_ignore` in it, the program crashes. I wrote this patch to fix it.

----

# Run the program

- Your working directory
    - data
        - train
        - test
	- config
        - your configuration file
    - mmdetection_patch
        - loading.py
	- fcos_train.py
    - evaluation.py
    - dump_result_to_json.py
- Unzip SVHN dataset under `data` directory.
- Clone `mmdetection` from its github. Replace `mmdet/datasets/pipelines/loading.py` to the patch file in `mmdetection_patch`.
- Install `mmdetection`. There's a tutorial for the installation.
- Modify the path of the checkpoint and configuration file in `fcos_train.py`.
- Run `fcos_train.py`.
    ```python=
    $ python fcos_train.py
    ```
- You can run `evaluation.py` and `dump_result_to_json.py` with a similar command as above. You should notice that the path to the checkpoint and configuration file need to be change, just likes what we've done in `fcos_train.py`.
