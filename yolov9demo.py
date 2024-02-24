# E. Culurciello
# YOLO v9 object detector demo

# from: https://github.com/kadirnar/yolov9-pip (pip install yolov9pip)
# download model from: https://github.com/WongKinYiu/yolov9?tab=readme-ov-file

# export OMP_NUM_THREADS=16

import numpy as np
import yolov9
import cv2
import time

def generate_colors(n): 
  rgb_values = []
  r,g,b = 0,50,100
  step = 256 / n 
  for _ in range(n): 
    r += step 
    g += step 
    b += step 
    r = int(r) % 256 
    g = int(g) % 256 
    b = int(b) % 256 
    rgb_values.append((r,g,b)) 
  return rgb_values

# load pretrained or custom model
model = yolov9.load(
    "yolov9-c.pt",
    device="cpu",
)

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = None  # (optional list) filter by class
# print(model.names, model.names[1]) # categories names
colors = generate_colors(len(model.names))

# set camera capture:
camera_id = 0
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(camera_id)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # perform inference
    t = time.time()
    results = model(frame)
    # results = model(frame, size=frameWidth)
    t = time.time() - t

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    bboxes = np.array(boxes, dtype="int")
    classes = np.array(categories, dtype="int")

    # show detection bounding boxes on image
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), colors[cls], 2)
        cv2.putText(frame, model.names[cls], (x, y - 5), 
                    cv2.FONT_HERSHEY_PLAIN, 2, colors[cls], 2)
        cv2.putText(frame, "FPS: "+str(1/t), (10, 30), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)

    # results.show()
    cv2.imshow("Img", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()