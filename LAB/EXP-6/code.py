import cv2

# Open the same video twice
cap_slow = cv2.VideoCapture("video.mp4")
cap_fast = cv2.VideoCapture("video.mp4")

while True:
    # Read one frame for slow motion
    ret1, frame1 = cap_slow.read()

    # Read one frame for fast motion
    ret2, frame2 = cap_fast.read()

    if not ret1 or not ret2:
        break

    # Display both videos
    cv2.imshow("Slow Motion", frame1)
    cv2.imshow("Fast Motion", frame2)

    # Skip 2 frames for fast motion
    cap_fast.read()
    cap_fast.read()

    # Delay for slow motion
    if cv2.waitKey(100) & 0xFF == ord(' '):
        break

# Release resources
cap_slow.release()
cap_fast.release()
cv2.destroyAllWindows()
