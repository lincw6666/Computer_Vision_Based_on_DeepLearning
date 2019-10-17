DLCV Homework 01
===

# Files description

- **DLCV_HW01.ipyn0b**: I done everythings on the google CoLab since I don't have a GPU. This jupyter notebook is what I'd worked on the CoLab. So there're some blocks such as mounting google drive, which is useless when you run locally.
- **dlcv_hw01.py**: This is just for satisfy the homework requirement. The codes inside satisfiy the PEP8 coding style.
- **hw1_net.pth**: It contains all the parameters of my trained CNN model.
- **result.csv**: It stores the predictions on unlabeled data. And this is the file I upload to Kaggle.

----

# Directory tree for running the program

- Your working directory
    - dataset
        - dataset
            - train
            - test
    - hw1_net.pth (during runtime)
    - result.csv (during runtime)
- The `dataset` directory is what you unzip from `cs-ioc5008-hw1.zip`.
- If you put your data in google drive, you need to modify the `Unzip it to your home directory` block in *DLCV_HW01.ipynb*.
