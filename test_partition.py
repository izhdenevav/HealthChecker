import test_dlib
import cv2

video = './subject33/vid.avi'

cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = test_dlib.get_landmarks_dlib(frame)

    for i in range(68):
        x, y = landmarks[i].x, landmarks[i].y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow("Face Landmarks", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

