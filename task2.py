from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # load a pretrained model

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # open the webcam

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()


while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model.predict(source=frame, classes=[0,67], verbose=False) #predict on the webcam frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # get the bounding box coordinates
        conf = float(box.conf[0])  # get the confidence score
        cls = int(box.cls[0])  # get the class index
        if cls == 0:
            color = (255, 0, 0)  # blue for person
            name = 'Person'

        else:
            color = (0,0,255)  # red for cell phone
            name = 'Cell Phone'
        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # display the frame with the detected objects
    cv2.imshow("YOLOv8 Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #exit the loop if the window is closed
    if cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cam.release()  # release the webcam
cv2.destroyAllWindows()  # close all OpenCV windows

