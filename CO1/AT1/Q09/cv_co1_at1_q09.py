import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
print("Upload an image containing one or more faces")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------
# Low-Level Vision
# Image Enhancement
# ---------------------------------------

enhanced = cv2.equalizeHist(gray)

# ---------------------------------------
# Mid-Level Vision
# Face Detection
# ---------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

faces = face_cascade.detectMultiScale(
    enhanced,
    scaleFactor=1.1,
    minNeighbors=5
)

# Draw rectangles
output = img_rgb.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 2)

# ---------------------------------------
# Display
# ---------------------------------------

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(output)
plt.title("Detected Faces")
plt.axis("off")

plt.show()

# ---------------------------------------
# High-Level Vision
# ---------------------------------------

print("Computer Vision Levels in Facial Recognition\n")

print("1. Low-Level Vision")
print("   - Image acquisition")
print("   - Histogram Equalization")
print("   - Noise reduction")

print("\n2. Mid-Level Vision")
print("   - Face detection")
print("   - Feature extraction")

print("\n3. High-Level Vision")
print("   - Face recognition")
print("   - Identity verification")
print("   - Decision making")

print(f"\nFaces Detected: {len(faces)}")
