import dlib
import cv2
from collections import namedtuple

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks_dlib(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks_object = landmark_predictor(gray, face)

    Point = namedtuple("Point", ["x", "y"])

    landmarks = {
        i: Point(point.x, point.y)
        for i, point in enumerate(landmarks_object.parts())
    }

    return landmarks

    

