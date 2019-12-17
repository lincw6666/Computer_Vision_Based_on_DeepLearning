# Install dependencies.
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2
pip install -e detectron2

# Prepare the dataset.
mkdir data
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-3pTCXBCCZDCzs-ARSpgkP9L7tetWzY2' -O data/pascal_train.json
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-8LgT4DH8VdZXtyH9NCI_UA4vRA5Acs-' -O data/test.json
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-BbVj_ZwsxOnY043ZcqIY2IJVtYbl7RX' -O data/train_images.zip
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-9thvdluXGxE_Shi83dsp9XMIjgbE9ZQ' -O data/test_images.zip
unzip -q -d data data/train_images.zip
unzip -q -d data data/test_images.zip

# Download the weight of my pre-trained model.
mkdir model_weight
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=121v5UJ8ga02qgNC2bnxlF1-PRdtVcv4T' -O model_weight/model.pth

# Prepare the config file.
cp configs/mask_rcnn_R_50_FPN_decay_0.01_momentum_0.9.yaml detectron2/configs/COCO-InstanceSegmentation/