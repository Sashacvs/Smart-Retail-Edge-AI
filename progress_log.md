# Progress Log

## Stage 0 - Setup
- Created project folder and virtual environment
- Installed OpenCV, YOLO (Ultralytics), numpy, pandas
- Verified setup using test script

## Stage 1 - Object Detection
- Implemented real-time object detection using YOLO
- Used webcam feed with OpenCV
- Displayed bounding boxes and confidence scores
- Observed some incorrect detections (expected at this stage)

## Stage 2 - Face Detection (Haar Cascade)
- Implemented basic face detection using Haar Cascade
- Observed major issues:
  - false positives
  - unstable detection
  - poor performance on movement and angles

## Stage 3 - Face Detection (YuNet Upgrade)
- Switched to YuNet (OpenCV DNN face detector)
- Significant improvement in detection quality
- Face box more stable and consistent
- Keypoints detected (eyes, nose, mouth)
- Detection works during movement and slight side angles

Next:
- Implement face tracking (following face across frames)
- Face detection with YuNet is working reliably
- While moving to face tracking, encountered OpenCV build limitation: CSRT tracker was not available in the current install
- Next action: switch from opencv-python to opencv-contrib-python to enable tracking modules

## Stage 4 - Face Detection + Tracking

- Successfully implemented face detection using YuNet (OpenCV DNN)
- Added face tracking using CSRT tracker
- Able to lock onto a face and follow it across frames
- Tracking works well during normal movement

Observations:
- Tracker may drift when face is blurred or under strong lighting changes
- Requires manual reset in such cases

Conclusion:
- Requirement 1 (face detection + following) completed at prototype level

Next:
- Improve robustness (re-detection when tracking fails)
- Move to individual distinction (assigning IDs)

## Requirement 2 - Phase 1 (Multi-Face Detection)

- Created `multi_face_tracking.py`
- Used YuNet to detect multiple faces in real time
- Added bounding boxes and temporary IDs for each detected face
- Displayed face confidence scores on screen

Observations:
- Multi-face detection works
- IDs are currently assigned per frame only
- IDs are not yet stable across movement
- Next step is to maintain identity across frames