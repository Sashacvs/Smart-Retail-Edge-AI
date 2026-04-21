import cv2

# Load YuNet model
model_path = "models/face_detection_yunet_2023mar.onnx"

cap = cv2.VideoCapture(0)

# Read one frame to get size
ret, frame = cap.read()
h, w, _ = frame.shape

# Create detector
detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update input size (important for dynamic frames)
    detector.setInputSize((frame.shape[1], frame.shape[0]))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:
        for i, face in enumerate(faces):
            x, y, w, h = map(int, face[:4])
            score = face[-1]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw ID
            cv2.putText(
                frame,
                f"ID {i+1} ({score:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    cv2.imshow("Multi Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()