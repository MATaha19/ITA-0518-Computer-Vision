import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload document image
print("Upload an image containing text")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------
# Low Sampling (Low Resolution)
# ---------------------------------------

low_res = cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4))
low_res = cv2.resize(low_res,
                     (gray.shape[1], gray.shape[0]),
                     interpolation=cv2.INTER_NEAREST)

# ---------------------------------------
# Low Quantization (16 Gray Levels)
# ---------------------------------------

quantized = (gray // 16) * 16

# ---------------------------------------
# Improve Acquisition using Thresholding
# ---------------------------------------

_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---------------------------------------
# Display
# ---------------------------------------

plt.figure(figsize=(16,5))

plt.subplot(1,4,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(low_res, cmap='gray')
plt.title("Low Sampling")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(quantized, cmap='gray')
plt.title("Low Quantization")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(binary, cmap='gray')
plt.title("Improved for OCR")
plt.axis("off")

plt.show()

# ---------------------------------------
# Explanation
# ---------------------------------------

print("OCR Performance Factors")
print("-----------------------")
print("1. Image Acquisition : Clear images improve OCR accuracy.")
print("2. Sampling          : Higher resolution preserves character shapes.")
print("3. Quantization      : More gray levels retain text details.")
print("4. Thresholding      : Improves text-background separation.")
