import cv2
import numpy as np

#access the camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()
else:
    print("Camera opened successfully.")

# define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8) 

#read a frame from the camera

while True:
    ret, frame = cam.read()

    if not ret:
        print("Error: Could not read frame from camera.")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Frame", hsv_frame)

    # define lower and upper HSV boundaries for the color blue
    lowerBlue = np.array([100, 150, 50])
    upperBlue = np.array([140, 255, 255])

    # define lower and upper HSV boundaries for the color red
    lowerRed = np.array([0, 150, 50])
    upperRed = np.array([10, 255, 255])

    lowerRed2 = np.array([170, 150, 50])
    upperRed2 = np.array([179, 255, 255])


    blueMask = cv2.inRange(hsv_frame, lowerBlue, upperBlue)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_CLOSE, kernel)

    redMask1 = cv2.inRange(hsv_frame, lowerRed, upperRed)
    redMask2 = cv2.inRange(hsv_frame, lowerRed2, upperRed2)
    redMask = cv2.bitwise_or(redMask1, redMask2)
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_CLOSE, kernel)

    #contours for blue color
    contoursBlue, _ = cv2.findContours(blueMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blueObjectCount = 0
    for contour in contoursBlue:
        area = cv2.contourArea(contour)
        if area > 500:
            blueObjectCount += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Blue Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    #contours for red color
    contoursRed, _ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    redObjectCount = 0
    for contour in contoursRed:
        area = cv2.contourArea(contour)
        if area > 500:
            redObjectCount += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #display masks
    puttext = f"Blue Objects: {blueObjectCount}, Red Objects: {redObjectCount}"
    cv2.putText(frame, puttext, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Camera Frame", frame)
    cv2.imshow("Blue Mask", blueMask)
    cv2.imshow("Red Mask", redMask)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

