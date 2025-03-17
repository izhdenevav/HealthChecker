import test_dlib
import cv2

video = './subject33/vid.avi'

cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = test_dlib.get_landmarks_dlib(frame)

    print(landmarks)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

