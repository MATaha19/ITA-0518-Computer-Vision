import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload grayscale image
print("Upload a grayscale image")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# -------------------------------
# Different Quantization Levels
# -------------------------------

# 8-bit (256 levels)
img_8bit = img.copy()

# 4-bit (16 levels)
img_4bit = (img // 16) * 16

# 2-bit (4 levels)
img_2bit = (img // 64) * 64

# -------------------------------
# Display
# -------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img_8bit, cmap='gray')
plt.title("8-bit (256 Levels)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_4bit, cmap='gray')
plt.title("4-bit (16 Levels)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_2bit, cmap='gray')
plt.title("2-bit (4 Levels)")
plt.axis("off")

plt.show()

# -------------------------------
# Explanation
# -------------------------------

print("Observation:")
print("- 8-bit image preserves fine details.")
print("- 4-bit image loses some gray-level information.")
print("- 2-bit image has very few intensity levels and shows banding.")
print("\nConclusion:")
print("Higher quantization levels preserve more image detail, while lower levels reduce image quality.")
