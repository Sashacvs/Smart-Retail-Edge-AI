import cv2

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

# -----------------------------
# Helper: create tracker safely
# -----------------------------
def create_tracker():
    # Try legacy tracker first
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    # Fallback: direct tracker
    elif hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    else:
        raise AttributeError("CSRT tracker is not available in your OpenCV build.")

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Read first frame to get resolution
ret, frame = cap.read()
if not ret:
    print("Failed to read initial frame")
    cap.release()
    exit()

frame_h, frame_w = frame.shape[:2]

# -----------------------------
# Create YuNet detector
# -----------------------------
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (frame_w, frame_h),
    score_threshold=0.85,
    nms_threshold=0.3,
    top_k=5000
)

# -----------------------------
# Tracking state
# -----------------------------
tracker = None
tracking = False
tracked_box = None

print("Press 's' to detect and start tracking the main face.")
print("Press 'r' to reset tracking.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # Start detection + tracker init
    # -----------------------------
    if key == ord("s"):
        _, faces = detector.detect(frame)

        if faces is not None and len(faces) > 0:
            # Choose the face with the highest score
            best_face = max(faces, key=lambda f: f[-1])

            x, y, width, height = best_face[:4].astype(int)
            tracked_box = (x, y, width, height)

            tracker = create_tracker()
            tracker.init(frame, tracked_box)
            tracking = True

            print("Tracking started.")

        else:
            print("No face detected to start tracking.")

    # -----------------------------
    # Reset tracking
    # -----------------------------
    if key == ord("r"):
        tracker = None
        tracking = False
        tracked_box = None
        print("Tracking reset.")

    # -----------------------------
    # If tracking is active, update tracker
    # -----------------------------
    if tracking and tracker is not None:
        success, box = tracker.update(frame)

        if success:
            x, y, width, height = map(int, box)
            tracked_box = (x, y, width, height)

            # Draw tracked face box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Tracked Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        else:
            tracking = False
            tracker = None
            print("Tracking lost.")

    # -----------------------------
    # If not tracking, show detection preview
    # -----------------------------
    if not tracking:
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, width, height = face[:4].astype(int)
                score = face[-1]

                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"Face {score:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

    cv2.imshow("YuNet Face Tracking", frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()