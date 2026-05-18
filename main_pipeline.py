import cv2
import time
from core.detector import YuNetDetector
from core.tracker import CentroidTracker

def main():
    model_path = "models/face_detection_yunet_2023mar.onnx"
    detector = YuNetDetector(model_path)
    tracker = CentroidTracker(distance_threshold=80)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("[INFO] YuNet + Centroid Tracking Pipeline active. Press 'q' to exit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera Disconnected or frame not received.")
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        # Track faces
        tracked_faces = tracker.update(faces)

        # Draw tracked faces and IDs
        for x, y, w, h, track_id in tracked_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"ID {track_id}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time
        cv2.putText(
            frame, f"FPS: {int(fps)}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
        )

        cv2.imshow("Face Detection + Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()