import cv2

model_path = "models/face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))

cap = cv2.VideoCapture(0)

print("[INFO] AI Model loaded. FPS Counter Active. Press 'q' to exit.")

while True:
    # Start the clock for this specific frame
    timer = cv2.getTickCount()
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Camera Disconnected")
        break
    h, w, _ = frame.shape

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, face_w, face_h = map(int, face[:4])

            cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (0, 255, 0), 2)
    # Calculate Frames Per Second (FPS)
    # cv2.getTickFrequency() returns the number of clock-cycles per second
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw the FPS number on the top-left corner of the frame
    # cv2.putText(image, text, position, font, scale, color_bgr, thickness)
    cv2.putText(
        frame, f"FPS: {int(fps)}", (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
    )
    #cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 3)

    cv2.imshow("My raw video feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()