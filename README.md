# CS236G_Project

CycleGAN implementation for photo-to-painting image translation. 

BaselinGAN.ipynb: The baseline implementation is in BaselineGAN.ipynb. Cells in the notebook can be executed to import necessary dependencies, define the baseline model architecture and train using the initial data. Data used in BaselineGAN is located in the data folder in subdirectories trainA and trainB. 

FaceFilter.ipynb: Notebook containing the modified model architecture and code to run training can be found here. 

FaceFilterTraining.py: Running FaceFilterTraining.py runs the entire training process for the CycleGAN model, including progressive resizing. Note: this takes a long time to run! 

FacePreprocess.ipynb: Notebook containing code to preprocess the UTKFace images by upscaling with OpenCV. 

data: All data used for model training in the above code is located in subfolders trainA and trainB. 
