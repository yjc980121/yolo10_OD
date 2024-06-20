from ultralytics import YOLOv10 as YOLO
import streamlit as st
import cv2
import numpy as np
import settings
import matplotlib.pyplot as plt

def load_model(model_path):
    model = YOLO(model_path)
    return model

def _display_detected_frames(conf, model, st_frame, image):
    if isinstance(image, dict):
        st.error("Invalid image format: 'dict' object received.")
        return

    # Convert image to RGB format for processing with OpenCV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, conf=conf)

    # Get bounding boxes, labels, and confidences
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    # Category dictionary
    category_dict = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    # Initialize colors
    num_classes = len(category_dict)
    colors = plt.cm.get_cmap('hsv', num_classes)

    # Prepare annotations
    for box, label, confidence in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box.astype(int)
        label_name = category_dict[int(label)]
        confidence_text = f"{label_name} {confidence:.2f}"
        class_color = colors(int(label) / num_classes)[:3]
        class_color = [int(c * 255) for c in class_color]

        # Draw bounding boxes and labels on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), class_color, 2)
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), class_color, -1)
        cv2.putText(image, confidence_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Display the annotated image in Streamlit
    st_frame.image(image, caption='Detected Image', channels="RGB", use_column_width=True)

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    _display_detected_frames(conf, model, st_frame, image_rgb)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
