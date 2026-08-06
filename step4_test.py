import cv2
import easyocr

detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
reader = easyocr.Reader(['en'], gpu=False)   # first run downloads OCR model (~one time)

cap = cv2.VideoCapture("your_video.mp4")
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
plates = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 20))

for (x, y, w, h) in plates:
    plate_crop = frame[y:y+h, x:x+w]     # crop just the plate region
    results = reader.readtext(plate_crop)

    for (bbox, text, confidence) in results:
        print(f"Read: '{text}'  (confidence: {confidence:.2f})")
