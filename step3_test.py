import cv2

detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

cap = cv2.VideoCapture("your_video.mp4")
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # cascade works on grayscale

# detectMultiScale scans the image at multiple sizes to find plate-shaped regions
plates = detector.detectMultiScale(
    gray,
    scaleFactor=1.05,   # how much the image size is reduced at each scan
    minNeighbors=4,     # higher = fewer false positives, but may miss real plates
    minSize=(60, 20)    # ignore anything smaller than this (in pixels)
)

print(f"Found {len(plates)} candidate plate region(s)")

for (x, y, w, h) in plates:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite("detected_frame.jpg", frame)
print("Saved detected_frame.jpg — open it to check the boxes")
