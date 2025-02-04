import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# Load YOLO model from 'yolov4_model' directory
MODEL_PATH = "yolov4_model"
CFG_PATH = os.path.join(MODEL_PATH, "yolov4-custom.cfg")
WEIGHTS_PATH = os.path.join(MODEL_PATH, "yolov4-custom_best.weights")
NAMES_PATH = os.path.join(MODEL_PATH, "obj.names")

# Load YOLO network
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(NAMES_PATH, "r") as f:
    classes = f.read().strip().split("\n")

# Assign colors to each class
COLORS = [(0, 255, 0), (0, 0, 255), (0, 165, 255)]  # Green (mask), Red (no mask), Orange (incorrect)


# Function to detect masks in an image
def detect_mask(image):
    height, width = image.shape[:2]

    # Convert image to blob for YOLO processing
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]  # Class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes on the image
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            color = COLORS[class_ids[i]]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


# Webcam feed processing
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for better compatibility

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process frame through YOLO
        frame = detect_mask(frame)

        # Convert to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame for Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filename = "static/uploads/photo.jpg"
    file.save(filename)

    # Read image
    image = cv2.imread(filename)
    image = detect_mask(image)

    # Save processed image
    output_filename = "static/uploads/detected_photo.jpg"
    cv2.imwrite(output_filename, image)

    return render_template("index.html", uploaded_image=output_filename)


if __name__ == "__main__":
    app.run(debug=True)
