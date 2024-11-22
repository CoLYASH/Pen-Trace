from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import time
import imutils

# Set up argument parser to allow optional video input and buffer size
ap = argparse.ArgumentParser()
ap.add_argument("-v")  # Optional argument for video file input
ap.add_argument("-b", "--buffer", default=128, type=int, help="max buffer size")  # Argument for buffer size
args = vars(ap.parse_args())  # Parse arguments into a dictionary

# Define the HSV range for detecting white color
whiteLower = (0, 0, 200)
whiteUpper = (179, 30, 255)

# Deque to store tracked points, limited by the buffer size
pts = deque(maxlen=args["buffer"])

# Check if a video file is provided; if not, use the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()  # Start the webcam video stream
else:
    vs = cv2.VideoCapture(args["video"])  # Use the provided video file

# Allow the video stream or camera to warm up
time.sleep(2.0)

# Main loop for processing each frame
while True:
    # Read the current frame from the video stream
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame  # For video files, take the second return value
    if frame is None:
        break  # Exit loop if no frame is found (e.g., end of video)

    # Resize the frame to a width of 600 pixels for faster processing
    frame = imutils.resize(frame, width=600)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert the frame from BGR to HSV color space for color detection
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask to detect white objects within the specified HSV range
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)

    # Erode and dilate the mask to remove small noise and fill in gaps
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours (outlines) of the white object in the mask
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # Correct contour format for different OpenCV versions
    center = None

    # If any contours are found, process the largest one
    if len(cnts) > 0:
        # Find the contour with the largest area
        c = max(cnts, key=cv2.contourArea)

        # Calculate the minimum enclosing circle for the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Calculate the center of the contour using moments
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # If the detected object is large enough, draw the circle and center
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # Yellow circle
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red center point

    # Add the current center point to the deque of tracked points
    pts.append(center)

    # Loop over the tracked points to draw lines connecting them
    for i in range(1, len(pts)):
        # Skip if either point is None
        if pts[i - 1] is None or pts[i] is None:
            continue

        # Calculate line thickness based on the position in the deque (older points are thinner)
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)

        # Draw a red line between consecutive points
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Show the processed frame with the tracked object and path
    cv2.imshow("Frame", frame)

    # Check for 'q' key press to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video stream or video file
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
