import cv2

# Path to the YuNet face detection model
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Read one frame first to get webcam resolution
ret, frame = cap.read()
if not ret:
    print("Failed to read initial frame")
    cap.release()
    exit()

frame_h, frame_w = frame.shape[:2]

# Create YuNet face detector
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (frame_w, frame_h),
    score_threshold=0.85,
    nms_threshold=0.3,
    top_k=5000
)

print("YuNet face detection started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Update detector size to match current frame
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, width, height = face[:4].astype(int)
            score = face[-1]

            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Draw confidence score
            cv2.putText(
                frame,
                f"Face {score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw facial keypoints
            keypoints = face[4:14].astype(int)
            for i in range(0, len(keypoints), 2):
                cv2.circle(frame, (keypoints[i], keypoints[i + 1]), 2, (0, 0, 255), -1)

    cv2.imshow("YuNet Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()