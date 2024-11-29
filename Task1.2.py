import cv2
import numpy as np

id_marker = 7  # ID of the marker you're looking for

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Set detector parameters
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.02
parameters.maxMarkerPerimeterRate = 4.0
parameters.perspectiveRemovePixelPerCell = 4
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementMaxIterations = 50
parameters.cornerRefinementMinAccuracy = 0.1

# Load the image to overlay
image_augment = cv2.imread("/Users/Work/Desktop/img.jpg")
if image_augment is None:
    print("Error: Could not load 'img.jpeg'. Check the file path.")
    exit()
else:
    print("Image loaded successfully.")

# Get the dimensions of the overlay image (in pixels)
image_height, image_width, _ = image_augment.shape

# Define the scaling factor for the overlay image to match the marker
scale_factor = 3  # Image should exceed marker size by 3 times

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()
else:
    print("Webcam opened successfully.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame from the webcam.")
        break

    # Convert the frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()  # Flatten the array of IDs
        if id_marker in ids:
            # Get the index of the marker
            index = np.where(ids == id_marker)[0][0]
            marker_corners = corners[index][0]  # Get the corners of the detected marker

            # Define the target points for perspective transformation
            marker_width = int(np.linalg.norm(marker_corners[0] - marker_corners[1]))
            marker_height = int(np.linalg.norm(marker_corners[1] - marker_corners[2]))

            overlay_width = marker_width * 4
            overlay_height = marker_height * 3

            resized_overlay = cv2.resize(image_augment, (overlay_width, overlay_height))

            # Compute the target corners for the resized overlay
            top_left = marker_corners[0]
            top_right = marker_corners[1]
            bottom_right = marker_corners[2]
            bottom_left = marker_corners[3]

            # Expand the overlay dimensions outward
            expanded_corners = np.array([
                [top_left[0] - (overlay_width - marker_width) / 2, top_left[1] - (overlay_height - marker_height) / 2],
                [top_right[0] + (overlay_width - marker_width) / 2,
                 top_right[1] - (overlay_height - marker_height) / 2],
                [bottom_right[0] + (overlay_width - marker_width) / 2,
                 bottom_right[1] + (overlay_height - marker_height) / 2],
                [bottom_left[0] - (overlay_width - marker_width) / 2,
                 bottom_left[1] + (overlay_height - marker_height) / 2],
            ], dtype=np.float32)

            # Perform perspective transformation
            src_points = np.array([
                [0, 0],
                [resized_overlay.shape[1] - 1, 0],
                [resized_overlay.shape[1] - 1, resized_overlay.shape[0] - 1],
                [0, resized_overlay.shape[0] - 1]
            ], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, expanded_corners)
            warped_overlay = cv2.warpPerspective(resized_overlay, matrix, (frame.shape[1], frame.shape[0]))

            # Create a mask for blending
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(expanded_corners)], (255, 255, 255))
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Invert the mask
            mask_inv = cv2.bitwise_not(mask_gray)

            # Blend the overlay with the original frame
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            overlay_fg = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask_gray)

            # Add the two images together
            frame = cv2.add(frame_bg, overlay_fg)

    # Display the frame with the overlay
    cv2.imshow('Input', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
