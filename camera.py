import cv2

# Открываем веб-камеру
cap = cv2.VideoCapture(0)

# Получаем ширину и высоту кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Разрешение веб-камеры: {width}x{height}")

# Закрываем камеру
cap.release()
