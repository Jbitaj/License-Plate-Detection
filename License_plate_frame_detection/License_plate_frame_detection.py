# import os
import numpy as np
# import pandas as pd
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import math

# Read the image file
image = cv.imread('1.jpg')
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

image = imutils.resize(image, width=500)
img=cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Display the original image
fig, ax = plt.subplots(2, 2, figsize=(10,7))
ax[0,0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
ax[0,0].set_title('Original Image')

# RGB to Gray scale conversion
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ax[0,1].imshow(gray, cmap='gray')
ax[0,1].set_title('Grayscale Conversion')

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv.GaussianBlur(gray, (5,5), 0)
ax[1,0].imshow(gray, cmap='gray')
ax[1,0].set_title('Bilateral Filter')

# Find Edges of the grayscale image
edged = cv.Canny(gray, 50, 190)
ax[1,1].imshow(edged, cmap='gray')
ax[1,1].set_title('Canny Edges')

fig.tight_layout()
plt.show()

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
edged = cv.dilate(edged, kernel, iterations=1)
edged = cv.erode(edged, kernel, iterations=1)

# Find contours based on Edges
cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:10] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
best_plate = None
best_score = 0  

for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4 and cv.isContourConvex(approx):  
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = w / float(h)

        # فیلتر کردن براساس اندازه و نسبت طول به عرض
        if 80 < w < 400 and 20 < h < 100 and 2.5 < aspect_ratio < 6:  
            area = w * h
            solidity = cv.contourArea(c) / float(area)

            # محاسبه‌ی امتیاز براساس اندازه و پرشدگی
            score = area * solidity  

            if score > best_score:  
                best_score = score
                best_plate = approx
                ROI = img[y:y+h, x:x+w]

# نمایش پلاک شناسایی‌شده
if best_plate is not None:
    cv.drawContours(image, [best_plate], -1, (0, 255, 0), 3)

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("License Plate Detection")
plt.show()
