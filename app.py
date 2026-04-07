import cv2
from ultralytics import YOLO

# Load YOLO model (brain)
model = YOLO("yolov8n.pt")

# Open webcam (camera input)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Starting live detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Run detection on the frame
    results = model(frame, verbose=False)

    # Loop through detections
    for r in results:
        for box in r.boxes:
            # Get class ID and confidence
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = model.names[cls_id]

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Show frame
    cv2.imshow("Smart Retail Edge AI - Live Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
