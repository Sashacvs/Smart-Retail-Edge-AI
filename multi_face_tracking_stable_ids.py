import cv2
import math

model_path = "models/face_detection_yunet_2023mar.onnx"

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
h, w, _ = frame.shape

detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (w, h)
)

previous_faces = []
next_id = 1
distance_threshold = 80

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)

    current_faces = []

    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            score = face[-1]

            center_x = x + w // 2
            center_y = y + h // 2

            matched_id = None
            min_distance = float("inf")

            for prev_face in previous_faces:
                prev_x, prev_y, prev_w, prev_h, prev_id = prev_face

                prev_center_x = prev_x + prev_w // 2
                prev_center_y = prev_y + prev_h // 2

                distance = math.hypot(center_x - prev_center_x, center_y - prev_center_y)

                if distance < min_distance and distance < distance_threshold:
                    min_distance = distance
                    matched_id = prev_id

            if matched_id is None:
                matched_id = next_id
                next_id += 1

            current_faces.append((x, y, w, h, matched_id))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {matched_id} ({score:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    previous_faces = current_faces

    cv2.imshow("Stable Face IDs", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()