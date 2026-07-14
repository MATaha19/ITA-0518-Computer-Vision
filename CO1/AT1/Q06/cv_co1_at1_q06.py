import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
print("Upload a road or driving image")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------
# Image Enhancement (Image Formation Model)
# ---------------------------------------

# Convert to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

# Split channels
l, a, b = cv2.split(lab)

# Apply CLAHE to improve brightness and contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l_enhanced = clahe.apply(l)

# Merge channels
enhanced_lab = cv2.merge((l_enhanced, a, b))

# Convert back to RGB
enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

# ---------------------------------------
# Display Images
# ---------------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(enhanced_img)
plt.title("Enhanced Image")
plt.axis("off")

plt.show()

# ---------------------------------------
# Explanation
# ---------------------------------------

print("\nImage Formation Model:")
print("- Improves image brightness and contrast.")
print("- Enhances visibility of roads and objects.")
print("- Reduces effects of poor lighting.")
print("- Helps detect lanes, vehicles, pedestrians, and traffic signs.")

print("\nApplication:")
print("Autonomous Driving")
