import math

class CentroidTracker:
    def __init__(self, distance_threshold=80):
        self.previous_faces = []  # Each entry: (x, y, w, h, id)
        self.next_id = 1
        self.distance_threshold = distance_threshold

    def update(self, detected_faces):
        """
        detected_faces: array-like with (x, y, w, h, ...) for each face (as returned by YuNet)
        Returns: list of (x, y, w, h, matched_id)
        """
        current_faces = []

        if detected_faces is not None:
            for face in detected_faces:
                x, y, w, h = map(int, face[:4])
                center_x = x + w // 2
                center_y = y + h // 2

                matched_id = None
                min_distance = float("inf")

                # Try to match with previous faces using centroid distance
                for prev_face in self.previous_faces:
                    prev_x, prev_y, prev_w, prev_h, prev_id = prev_face
                    prev_center_x = prev_x + prev_w // 2
                    prev_center_y = prev_y + prev_h // 2
                    distance = math.hypot(center_x - prev_center_x, center_y - prev_center_y)
                    if distance < min_distance and distance < self.distance_threshold:
                        min_distance = distance
                        matched_id = prev_id

                # Assign new ID if no match
                if matched_id is None:
                    matched_id = self.next_id
                    self.next_id += 1

                current_faces.append((x, y, w, h, matched_id))

        self.previous_faces = current_faces
        return current_faces