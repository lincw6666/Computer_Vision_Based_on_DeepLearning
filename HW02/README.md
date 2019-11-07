DLCV Homework 02
===

# Files description

- **DLCV_HW02.py**: See `Run the program` section below to understand how to run it.
- **hw2_netG.pth**: It contains the parameters for the generator.
- **hw2_netD.pth**: It contains the parameters for the discriminator.
- **DLCV_HW02_DCGAN.ipynb**: This jupyter notebook is what I'd done on the CoLab. So there're some blocks such as mounting google drive, which is useless when you run locally. However, if you just want to test the code, I recommand you to use DLCV_HW02.py. This file is for development.
- **DLCV_HW02_WGAN.ipynb**: The model I used in this notebook is Wasserstein GAN. I expected it to perform better, but it didn't. I think that I must had done something wrong. I left this file for future learning.

----

# Run the program

- Your working directory
    - data
        - img_align_celeba
	- Results
	- helper.py
- Unzip celeba.zip under `data` directory.
- The output images will be stored under the `Results` directory.
- Use the following command to run the code:
	```
	$ python DLCV_HW02.py
	```
- It'll produce hw2_netD.pth and hw2_netG.pth during the runtime. Only store the newest version, i.e. it'll replace the old hw2_netD.pth and hw2_netG.pth. If you want to save all the .pth files, you must move it to another directory periodicly. (I suggest you to write some simple python code that will move the .pth files automatically)
