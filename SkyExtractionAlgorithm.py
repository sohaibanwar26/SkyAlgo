import os
import random
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from scipy.signal import medfilt
from skimage import io
import time
from skimage.metrics import structural_similarity

"""######################## IMAGE COLLECTION AND CATEGORIZATION (Mean Pixel Intensity) ########################"""

# Appends all images in folders to a data dictionary

# folders_ = ['1093', '4795', '8438', '10870']
folders_ = ['1093']
categories = ['night', 'day']
size_dict = {}
data_dict = {}
path_dict = {}


# Get mean pixel intensities of pixels, images, and folders
def mean_pixel_intensity_image(image_matrix):
    return int(np.mean(np.array(image_matrix).reshape(-1, 1)))


def verify_image(img_file):
    try:
        io.imread(img_file)
    except:
        return False
    return True


start = time.time()

for fldr_ in folders_:
    img_paths = []
    for imgFile_ in sorted(glob.glob(f"{fldr_}/*.jpg")):
        verified_image = verify_image(imgFile_)
        if verified_image:
            img_paths.append(imgFile_)

    size_dict[fldr_] = len(img_paths)
    data_dict[fldr_] = [
        # [cv2.cvtColor(cv2.imread(img_file_), cv2.COLOR_BGR2GRAY) for img_file_ in cv_img if verify_image(img_file_)],
        [cv2.cvtColor(cv2.imread(img_file_), cv2.COLOR_BGR2GRAY) for img_file_ in img_paths],
        [img_file_ for img_file_ in img_paths]]

end_load_images = time.time()

print(f"Time taken to load all images: {end_load_images - start :.2f}s")

meanPI_all = {}  # Mean pixel intensities collected from all folders
for key_ in data_dict:
    meanPI_per_folder = []  # Mean pixel intensity of a folder
    for img_ in data_dict[key_][0]:
        meanPI_per_image = mean_pixel_intensity_image(img_)  # Mean pixel intensities of individual images
        meanPI_per_folder.append(meanPI_per_image)
    meanPI_all[key_] = int(np.mean(meanPI_per_folder))

meanPI_all_value = int(np.mean(list(meanPI_all.values())))

print(f"Mean Pixel Intensity (MPI) of individual data folders: {meanPI_all}")
print(f"Mean Pixel Intensity (MPI) of the entire data: {meanPI_all_value}")

end_mean_pixel_intensity = time.time()

print(f"Time taken to find image, folder, and overall mean pixel intensity of all images: {end_mean_pixel_intensity - end_load_images :.2f}s")

# Divide the images in every folder based on global mean pixel intensity value
divided_data = {}

for fldr_ in data_dict:
    night_img_list = []
    day_img_list = []
    for i, img_ in enumerate(data_dict[fldr_][0]):
        if mean_pixel_intensity_image(img_) > meanPI_all_value:
            day_img_list.append([img_, data_dict[fldr_][1][i]])
        else:
            night_img_list.append([img_, data_dict[fldr_][1][i]])

    divided_data[fldr_ + f"_{categories[0]}"] = night_img_list
    divided_data[fldr_ + f"_{categories[1]}"] = day_img_list


# # DEBUG: Check that all divided images are appended in the dictionary (and respective dictionary keys)
# for key_ in divided_data:
#     print(f"{key_} : {len(divided_data[key_])}")


divide_data_categories = time.time()
print(f"Time taken to to divide all images into Day/Night categories: {divide_data_categories - end_mean_pixel_intensity :.2f}s")


"""################################# SKY EXTRACTION #################################"""

# Laplacian and Morphology Method


def cal_skyline(mask_):
    h, w = mask_.shape
    for i in range(w):
        raw = mask_[:, i]
        after_median = medfilt(raw, 19)
        try:
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            if first_zero_index > 1:
                mask_[first_one_index:first_zero_index, i] = 1
                mask_[first_zero_index:, i] = 0
                mask_[:first_one_index, i] = 0

            # # Added logic to turn all pixels of a column to 0 if the top most and bottom most pixel value is 0/Black
            # if mask_[0, i] == 0 and mask_[-1, i] == 0:
            #     mask_[:, i] = 0
        except:
            continue
    return mask_


def SkyExtractor(image_category_, data_folder_, image_index_):
    image_folder_ = f'{data_folder_}_{image_category_}'
    image_path = divided_data[image_folder_][image_index_][1]
    image_ = divided_data[image_folder_][image_index_][0]
    print(f"Random Index: {random_idx}\nRandom Image Path: {image_path}")

    org_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # Original image to compare in the end
    img_gray = image_.copy()  # Deep copy to perform operations

    img_gray = cv2.blur(img_gray, (3, 3))
    img_gray = cv2.medianBlur(img_gray, 5)
    lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    gradient_mask = (lap < 6).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel)

    mask = cal_skyline(mask)

    after_img = cv2.bitwise_and(org_img, org_img, mask=mask)

    after_img = histogram_based_cutoff(after_img, image_category_)
    show_comparison(org_img, after_img)
    calculate_extraction_efficiency(mask, data_folder_)


def show_comparison(org_img_, after_img_):
    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    ax[0].imshow(org_img_)
    ax[0].set_title("ORIGINAL")
    ax[1].imshow(after_img_)
    ax[1].set_title("MASKED FOR EXTRACTION")
    fig.suptitle("IMAGE COMPARISON")
    plt.show()


def calculate_extraction_efficiency(mask_, ground_truth_):
    ground_truth_img = cv2.imread(f'{ground_truth_}.png', 0)
    mask = mask_.copy()

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(ground_truth_img, mask, full=True)
    print("Image Similarity Metric (Scikit-Image): {:.2f}%".format(score * 100))

    # Crude method to calculate different pixels percentage
    difference = cv2.absdiff(mask, ground_truth_img)
    num_diff = cv2.countNonZero(difference)
    # print(num_diff)   # DEBUG
    print(f"Naive percentage absolute difference: {round(num_diff/(mask.shape[0] * mask.shape[1])*100, 2)}%")


def histogram_based_cutoff(_image, _category):
    return_image = _image.copy()
    hist_ = cv2.calcHist([return_image], [0], None, [256], [1, 256])
    return_image = np.clip(_image, 1, 243)

    return return_image


# random.seed(42)
# Randomize target image for sky extraction
# random_category = random.choice(categories)
random_category = 'day'
random_data_folder = random.choice(folders_)
random_idx = 467
# random_idx = random.randint(0, len(divided_data[random_data_folder]))

SkyExtractor(random_category, random_data_folder, random_idx)


