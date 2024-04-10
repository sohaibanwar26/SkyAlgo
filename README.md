## Introduction
A Sky Extraction Algorithm that separates the sky region from a given image and then compares it to the ground truth. This can be very useful in several situations such as Unmanned Air Vehicles (UAV) navigating through areas with complex landscapes.

## Dataset
The dataset contains around 4000 images with 4 different landscapes. The images were taken in various weather conditions and at different times of the day to test the robustness of the algorithm.

Original Images for the landscapes
![Original](https://github.com/sohaibanwar26/SkyAlgo/blob/main/Original%20Image.png)

Images for landscapes with low pixel density (PPI)
![Low](https://github.com/sohaibanwar26/SkyAlgo/blob/main/Low%20Pixel%20Density.png)

Grounds Truth for each corresponding landscape:
![ground_truth](https://github.com/sohaibanwar26/SkyAlgo/blob/main/ground_truth.png)

## Algorithm Explanation
The proposed algorithm is based on categorizing the images from the given dataset into night and day images and then applying the sky extraction algorithm to detect the sky region. This differentiation between day and night images is performed by obtaining the global mean pixel intensity value for images and then dividing the images in the datasets based on that. To obtain accuracy, the algorithm will first calculate the mean pixel intensities from all the folders of the dataset, then the mean pixel intensity for each folder, and finally images in the dataset are divided based on global mean pixel intensity values. The function checks if the mean pixel intensity of the image is greater than the one of all folders, if yes it will automatically locate it in the day image list. After the images are divided into day and night lists, the cal_skyline and SkyExrtractor functions are called. The cal_skyline produces a mask and for SkyExtractor three main inputs are taken (image_category, data folder, image index). The original image is saved for comparison while gray version image is then used to perform the actions. The image is first blurred to reduce noise and improve accuracy and then a Laplacian filter is applied on the grayscale image. The purpose of the Laplacian filter is to determine if a change in adjacent pixel values is from an edge or continuous progression.

## Results
The algorithm selects a random image and compares the original image with the resulting image.
![result](https://github.com/sohaibanwar26/SkyAlgo/blob/main/Untitled_design.png)

To make the script more user-friendly and better determine the efficiency of the algorithm, the following information/metrics are implemented:

* Time taken to load all images

* Mean Pixel Intensity (MPI) of individual data folders

* Mean Pixel Intensity (MPI) of the entire data

* Time taken to find image, folder, and overall mean pixel intensity of all images

* Time is taken to divide all images into Day/Night categories

* Random Index Value

* Random Image Path


## Demo




## Running on Local PC

Dependencies:

    To run this project you can install the following dependencies through pip in the command line:

    NumPy:  pip install numpy
    OpenCV: pip install opencv
    Scipy:  python -m pip install scipy
    Skimage: python -m pip install -U scikit-image
    


Installation:

    Select an IDE (E.g Visual Studio, PyCharm) 
    Run the SkyExtractionAlgorithm.py script

Modifying:
    
    To modify the script according to your needs:
    
    Place the SkyExtractionAlgorithm.py file at the top-level directory where your image folders are located
    Rename the image folders in the script as per your dataset
    Run the SkyExtractionAlgorithm.py script
    


    
