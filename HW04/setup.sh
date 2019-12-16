# Install dependencies.
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2
pip install -e detectron2

# # Prepare the dataset.
unzip -q -d data data/train_images.zip
unzip -q -d data data/test_images.zip

# Prepare the config file.
cp configs/mask_rcnn_R_50_FPN_decay_0.01_momentum_0.9.yaml detectron2/configs/COCO-InstanceSegmentation/