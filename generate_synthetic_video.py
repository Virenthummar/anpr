import cv2
import numpy as np

width, height = 640, 480
out = cv2.VideoWriter("test_input.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))

for i in range(60):
    frame = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.rectangle(frame, (150, 150), (490, 350), (80, 80, 80), -1)      # fake car body
    px = 220 + i
    cv2.rectangle(frame, (px, 280), (px+200, 330), (255, 255, 255), -1)  # plate background
    cv2.rectangle(frame, (px, 280), (px+200, 330), (0, 0, 0), 2)         # plate border
    cv2.putText(frame, "GJ01AB1234", (px+10, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    out.write(frame)

out.release()
