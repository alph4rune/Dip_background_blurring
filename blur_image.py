import cv2
import numpy as np

# Load the image
image = cv2.imread(r"Path for the image")

# Load YOLOv3 weights and configuration
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO labels
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Detect objects in the image
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Filter out human detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:  # 0 corresponds to 'person' class in COCO dataset
            # Object detected is a person
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Blur only the background regions
blurred = cv2.GaussianBlur(image, (25, 25), 0)
for (x, y, w, h) in boxes:
    blurred[y:y+h, x:x+w] = image[y:y+h, x:x+w]

# Display the result
cv2.imshow('Result', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()