import math
import cv2
import numpy as np
import random
from collections import deque

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Deque to store center points of the detected object
center_points = deque()

while True:
    # Read a frame from the video capture
    _, frame = cap.read()

    # Flip the frame horizontally to provide a mirror image
    frame = cv2.flip(frame, 1)

    # Apply Gaussian blur to reduce noise
    blur_frame = cv2.GaussianBlur(frame, (7, 7), 0)

    # Convert the blurred frame to HSV color space for better color detection
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV values for the color blue
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a mask that isolates only the blue color in the frame
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Define a kernel for morphological operations (used to clean the mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Use morphological opening to remove small noise from the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask to detect shapes of the isolated blue object
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # If at least one contour is found
    if len(contours) > 0:
        # Find the largest contour by area (assumed to be the object we are tracking)
        biggest_contour = max(contours, key=cv2.contourArea)

        # Compute the moments of the largest contour to find its center
        moments = cv2.moments(biggest_contour)
        centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        # Draw a red circle at the center of the contour
        cv2.circle(frame, centre_of_contour, 5, (0, 0, 255), -1)

        # Fit an ellipse around the contour and draw it in yellow
        ellipse = cv2.fitEllipse(biggest_contour)
        cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

        # Add the center of the contour to the deque of center points
        center_points.appendleft(centre_of_contour)

    # Loop through the center points to draw a trail (lines between consecutive points)
    for i in range(1, len(center_points)):
        # Generate random colors for the lines
        b = random.randint(230, 255)
        g = random.randint(100, 255)
        r = random.randint(100, 255)

        # Draw the line only if the distance between consecutive points is small (<= 50 pixels)
        if math.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) +
                     ((center_points[i - 1][1] - center_points[i][1]) ** 2)) <= 50:
            # Draw a line between the two points with a random color and thickness of 4
            cv2.line(frame, center_points[i - 1], center_points[i], (b, g, r), 4)

    # Show the processed frame with the drawn contours and lines
    cv2.imshow('original', frame)

    # Wait for 5 milliseconds and check if the 'ESC' key is pressed (ASCII 27)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Release the video capture and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()
