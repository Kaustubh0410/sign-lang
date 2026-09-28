import cv2
import numpy as np
from tempfile import TemporaryDirectory

async def extract_frames(file):
    # Read uploaded file into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frames.append(buffer.tobytes())
    cap.release()
    return frames
