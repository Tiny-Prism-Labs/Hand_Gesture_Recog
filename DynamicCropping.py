import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL as Image

# We need to get all the paths for the images to later load them
imagepaths = []

# Go through all the files and subdirectories inside a folder and save path to images inside list
for root, dirs, files in os.walk(".", topdown=False): 
  for name in files:
    path = os.path.join(root, name)
    if path.endswith("png"): # We want only the images
      imagepaths.append(path)


i = 0
# We need to get all the paths for the images to later load them
for path in imagepaths:
    
    img = cv.imread(path)  # Reads image and returns np.array
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Converts into the correct colorspace (GRAY)

    # Apply thresholding to the grayscale image
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Find the contours in the image.
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour in the image.
    largest_contour = max(contours, key=cv.contourArea)

    # Get the bounding box of the largest contour.
    x, y, w, h = cv.boundingRect(largest_contour)

    # Crop the image around the bounding box.
    cropped_image = img[y:y + h, x:x + w]

    # Get the base filename without extension
    base_filename = os.path.basename(path)

    # Get the base name of the subdirectory
    subdirectory = os.path.basename(os.path.dirname(path))

    # Create a subdirectory if it doesn't exist
    cropped_subdirectory = os.path.join(r"C:\Users\MEHUL KINI\OneDrive\Desktop\Hand_Gesture_Recog\Cropped images",
                                        "cropped_" + subdirectory)
    os.makedirs(cropped_subdirectory, exist_ok=True)

    # Save the cropped image with a more descriptive filename in the subdirectory
    save_path = os.path.join(cropped_subdirectory, "cropped_%s_%i.png" % (subdirectory, i))
    cv.imwrite(save_path, cropped_image)

    i += 1