import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload satellite image
print("Upload a satellite image")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------
# Simulate Low Spatial Resolution
# ---------------------------------------

low_res = cv2.resize(
    img,
    (img.shape[1]//6, img.shape[0]//6),
    interpolation=cv2.INTER_AREA
)

low_res = cv2.resize(
    low_res,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# ---------------------------------------
# Enhance Contrast
# ---------------------------------------

lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)

enhanced = cv2.merge((l, a, b))
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

# ---------------------------------------
# Display
# ---------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Satellite Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(low_res)
plt.title("Low Resolution Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(enhanced)
plt.title("Enhanced Image")
plt.axis("off")

plt.show()

# ---------------------------------------
# Explanation
# ---------------------------------------

print("Digital Image Fundamentals")
print("- Higher sampling captures finer land details.")
print("- Higher quantization improves intensity differences.")
print("- Better resolution improves land classification.")

print("\nImage Acquisition")
print("- High-quality satellite sensors reduce noise.")
print("- Proper lighting improves visibility.")
print("- Good acquisition increases analysis accuracy.")

print("\nApplications")
print("- Agriculture")
print("- Urban Planning")
print("- Forest Monitoring")
print("- Disaster Management")
