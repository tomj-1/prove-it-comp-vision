# Prove-it-comp-vision

Prove-it-comp-vision is a computer vision project that uses YOLOPv2 to detect vehicles and lane lines from road images.

## Features

- Vehicle detection
- Lane line detection
- Object bounding boxes
- Lane line visualization

## How it works

Object Detection:

1. Road image is resized and preprocessed.
2. Image is sent through the YOLOPv2 model.
3. YOLOPv2 detects vehicles in the image.
4. Bounding boxes are drawn around detected vehicles.

Lane Detection:

1. Road image is resized and preprocessed.
2. Image is sent through the YOLOPv2 model.
3. YOLOPv2 generates a lane segmentation mask.
4. Lane mask is resized to the original image size.
5. Detected lane lines are highlighted on the image.

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy

## Setup

Clone the repository:

```bash
git clone YOUR_REPO_URL
cd YOUR_REPO_NAME
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Clone YOLOPv2:

```bash
git clone https://github.com/CAIC-AD/YOLOPv2.git
```

Download the YOLOPv2 model weights and place them in the weights folder.

## Run Program

Run the program with an image and choose either CPU or GPU.

Using CPU:

```bash
python Main.py --input "path/to/image.jpg" --device cpu
```

Using GPU:

```bash
python Main.py --input "path/to/image.jpg" --device gpu
```
