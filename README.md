# 🏥 Detection of Availability of Beds in Hospital by checking through the CCTV footage using YOLOv8

### 📚 Overview

- This project implements a hospital bed occupancy detection system using [YOLOv8](https://yolov8.com/) object detection system.
- It detects both hospital beds and persons, and uses an IoU (Intersection over Union) function to determine if a bed is occupied or not.
- Then it updates the availability of beds in hospital database in real time.
- The system can work on images (JPG/PNG) and videos (MP4/MKV), and is optimized for real-time video analysis.

---

### 🚀 Why YOLO?

- [YOLO](https://arxiv.org/abs/1506.02640) (You Only Look Once) is a state-of-the-art real-time object detection algorithm.

- It is fast, accurate, and lightweight — perfect for scenarios like hospitals where real-time monitoring is critical.

- Compared to traditional CNN-based object detection, YOLO brings:
    - Faster inference
    - High detection accuracy
    - Better performance even with fewer datasets
    - Simpler deployment on CPU or GPU
 
---

### 🤍 What YOLO Version?

- YOLOv8m.pt (medium model) for person detection (from Ultralytics pretrained COCO model) (COCOMO)

- Custom trained YOLOv8 model (best.pt) for hospital bed detection.

Training Details for Bed Detector:

- Framework: Ultralytics YOLOv8

- Input: ~2000 images of hospital beds (manually custom annotated)

- Output: best.pt model file after training epochs

---

### 🧹 How Detection Happens

- Step 1: Detect all hospital beds in a frame using the custom YOLOv8 bed model.

- Step 2: Detect all persons in the same frame using the YOLOv8 COCO pretrained model.

- Step 3: For each detected person:

Check if they are standing (ignored if height > 1.8 × width).

For lying persons, calculate IoU with each detected bed.

Assign the person to the bed with the highest IoU (if IoU ≥ 0.08).

Step 4: Count all assigned beds → these are occupied beds.

Free Beds = Total Beds - Occupied Beds

🔠 How Matching (Assignment) Works

IoU (Intersection over Union) is calculated between:

Person bounding box

Bed bounding box

IoU Formula:
'IoU = Area of Overlap / Area of Union'

A threshold of 0.08 is set — if IoU ≥ 0.08 → person is assigned to that bed.

This approach ensures:

Only lying persons are considered

Only meaningful overlaps are assigned

📈 Matching Accuracy

Matching Rate = (Correctly assigned beds) ÷ (Total beds detected)

During testing, typical matching accuracy is around 85–90% depending on:

Camera angle

Bed layout

Quality of training data

Errors mainly occur when:

Person is partially lying

Bed is occluded or outside frame

🎮 Real-time Video Detection

Video frames are processed using YOLOv8 at resized 640×480 resolution.

Multi-threading is used to detect on every 2–3 frames, achieving near 30 FPS real-time.

Output visualizes:

Blue boxes → detected beds

Green boxes → detected persons

Red boxes → occupied beds

Free bed count and occupancy details are shown live.

⚖️ Possible Optimizations (Future)

Switch to YOLOv8n (nano) for ultra-smooth 60 FPS detection.

TensorRT acceleration for faster GPU inference.

Track persons over time (DeepSORT tracking) to reduce detection overhead.

Model quantization (FP16, INT8) to further speed up predictions.

🛠️ How To Run

Install dependencies:
'pip install ultralytics opencv-python numpy'

Initialize models:
'person_model = YOLO("yolov8m.pt")'
'bed_model = YOLO("D:/runs/detect/train3/weights/best.pt")'

Run on an image:
'detect_bed_occupancy("sample_image.jpg")'

Run on a video:
'detect_bed_occupancy_realtime("sample_video.mp4")'

Press 'q' to exit visualization.

📂 Project Structure

train/ → YOLO training images and labels for bed detector

weights/ → trained best.pt model

utils/ → helper functions (IoU calculation, frame processing)

main.py → entry point for detection on images/videos

🔥 Key Highlights

Dual YOLO models working together

Custom IoU based bed-person matching logic

Optimized for real-time video applications

Modular, extendable detection pipeline

✨ Final Thought

This project showcases how deep learning and object detection can be combined to solve real-world problems like hospital bed management — making healthcare faster, smarter, and more efficient.


