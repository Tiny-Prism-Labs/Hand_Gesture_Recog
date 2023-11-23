import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL as Image

# We need to get all the paths for the images to later load them
imagepaths = []

i=0 #to keep a count of the images

# Go through all the files and subdirectories inside a folder and save path to images inside list
for root, dirs, files in os.walk(".", topdown=False): 
  for name in files:
    path = os.path.join(root, name)
    if path.endswith("png"): # We want only the images
      imagepaths.append(path)

print(len(imagepaths)) # If > 0, then a PNG image was loaded

for path in imagepaths:
  img = cv.imread(path) # Reads image and returns np.array
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
  _, thresh = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

    # Find the contours in the image.
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour in the image.
  largest_contour = max(contours, key=cv.contourArea)

    # Get the bounding box of the largest contour.
  x, y, w, h = cv.boundingRect(largest_contour)

    # Crop the image around the bounding box.
  cropped_image = thresh[y:y+h, x:x+w]
  # Get the base filename without extension
  base_filename = os.path.basename(path)
    
    # Save the cropped image with a more descriptive filename
  save_path = r"C:\Users\MEHUL KINI\OneDrive\Desktop\Hand_Gesture_Recog\Cropped images\cropped_%s_%i.png" % (base_filename, i)
  cv.imwrite(save_path, cropped_image)
    
  i += 1