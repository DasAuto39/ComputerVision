import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/sign_detector2/weights/best.pt ")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction
    results = model(frame)

    # Annotate detections on the frame
    annotated = results[0].plot()

    cv2.imshow("Sign Language Detection", annotated)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
