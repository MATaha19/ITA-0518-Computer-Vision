import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload surveillance image
print("Upload an image containing people or vehicles")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------
# Image Preprocessing
# ---------------------------------------
gray = cv2.equalizeHist(gray)

# ---------------------------------------
# Object Detection (Face Detection Example)
# ---------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30)
)

# Draw bounding boxes
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
# Features of Computer Vision
# ---------------------------------------

print("Computer Vision Features in Automated Surveillance")
print("-------------------------------------------------")
print("1. Image Acquisition")
print("2. Image Enhancement")
print("3. Object Detection")
print("4. Object Tracking")
print("5. Face Recognition")
print("6. Decision Making and Alert Generation")

print(f"\nNumber of Faces Detected: {len(faces)}")
