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

- Input: ~375 images of hospital beds (manually custom annotated)

- Output: best.pt model file after training for 50 epochs on CUDA

---

### 🧹 How Detection Happens

- Step 1: Detect all hospital beds in a frame using our custom YOLOv8 bed model (Check Bed_Detection.ipynb file for further details)

- Step 2: Detect all persons in the same frame using the YOLOv8 COCO pretrained model.

- Step 3: For each detected person:

    * Check if they are standing (ignored if height > (1.8 × width)).

    * For lying persons, calculate IoU with each detected bed.

    * Assign the person to the bed with the highest IoU (if IoU ≥ 0.08).

- Step 4: Count all assigned beds → then count the availaible beds.

    * Free Beds = Total Beds - Occupied Beds

---

### 🔠 How Matching (Assignment) Works

- IoU (Intersection over Union) is calculated between:

    * Person bounding box

    * Bed bounding box

- IoU Formula:
`IoU = Area of Overlap / Area of Union`

- A threshold of 0.08 is set — if IoU ≥ 0.08 → person is assigned to that bed.

This approach ensures:

- Only lying persons are considered

- Only meaningful overlaps are assigned

---

### 📈 Matching Accuracy

- Matching Rate = (Correctly assigned beds) ÷ (Total beds detected)

- During testing, typical matching accuracy is around 78–87% depending on:

    * Camera angle

    * Bad bed layout

    * Quality of input media file
 
    * less amount of media in training dataset 

- Errors mainly occur when:

    * Bed is too far from the camera.

    * Bed is occluded or outside frame.
 
    * Person standing in front of camera, shadowing the patient and as a result, the patient doesn't get detected.

---

### 🎮 Real-time Video Detection

- Video frames are processed using YOLOv8 at resized 640×480 resolution.

- Output visualizes:

    * Blue boxes → detected beds

    * Green boxes → detected persons

    * Red boxes → occupied beds

Free bed count and occupancy details are shown live and updated on the database in real time.

---

### ⚖️ Possible Optimizations (Future)

- Switch to YOLOv8n (nano) for ultra-smooth 60 FPS detection.

- TensorRT acceleration for faster GPU inference.

- Track persons over time (DeepSORT tracking) to reduce detection overhead.

- Model quantization (FP16, INT8) to further speed up predictions.

---

### 🛠️ How To Run

1. Installing the dependencies:
```python
pip install ultralytics opencv-python numpy
```

2. Initialize models:
```python
person_model = YOLO("yolov8m.pt")
```
```python
bed_model = YOLO("D:/runs/detect/train3/weights/best.pt")  # This is my runs folder, your's could be different
```

3. Run on an image:
```python
detect_bed_occupancy("sample_image.jpg")
```

4. Run on a video:
```python
detect_bed_occupancy_realtime("sample_video.mp4")
```

Press 'q' to exit visualization.

---

### 🔥 Key Highlights

- Dual YOLO models working together synchronously.

- Custom trained Bed Detection model trained on 375+ bed image dataset for 50 epochs.

- Custom IoU based bed-person matching logic for optimised output.

- Takes both image and video files as input and process them simultaneously. (works with CCTV footagery)

- Modular, extendable detection pipeline.

---

### ✨ Final Thought

With the advent of this project, the necessity of manually handling availability of beds in hospitals and nursing homes is finally history. 

With automation like this, bed availability can be easily automated - making healthcare faster, smarter, and more efficient.

This project showcases how deep learning and object detection can be combined to solve real-world problems like hospital bed management.

##### Made with 💙 and late night ☕ by Sarthik
