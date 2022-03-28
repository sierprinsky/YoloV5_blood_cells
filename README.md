## <div align="center">YoloV5_blood_cells</div>

Overview: This is a dataset of blood cells photos, originally open sourced by cosmicad and akshaylambda. Use Cases: This is a small scale object detection dataset, commonly used to assess model performance. It's a example of medical imaging capabilities

## <div align="center">BCCD > raw</div>

https://public.roboflow.ai/object-detection/bccd

Provided by [Roboflow](https://roboflow.ai)
License: MIT

## <div align="center">Overview</div>
 
This is a dataset of blood cells photos, originally open sourced by [cosmicad](https://github.com/cosmicad/dataset) and [akshaylambda](https://github.com/akshaylamba/all_CELL_data). 

There are 364 images across three classes: `WBC` (white blood cells), `RBC` (red blood cells), and `Platelets`. There are 4888 labels across 3 classes (and 0 null examples).

Here's a class count from Roboflow's Dataset Health Check:

![BCCD health](https://i.imgur.com/BVopW9p.png)

And here's an example image:

![Blood Cell Example](https://i.imgur.com/QwyX2aD.png)

`Fork` this dataset (upper right hand corner) to receive the raw images, or (to save space) grab the 500x500 export.

## <div align="center">Use Cases</div>

This is a small scale object detection dataset, commonly used to assess model performance. It's a first example of medical imaging capabilities.

## <div align="center">Using this Dataset</div>

We're releasing the data as public domain. Feel free to use it for any purpose.

It's not required to provide attribution, but it'd be nice! :)

## <div align="center">About Roboflow</div>

[Roboflow](https://roboflow.ai) makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless. Developers reduce 50% of their boilerplate code when using Roboflow's workflow, automate annotation quality assurance, save training time, and increase model reproducibility. 

---------------------------------------------------------------------------------------------------------

## <div align="center">YOLOv5 Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Inference</summary>

Inference with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
. [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>



<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://github.com/ultralytics/yolov5/issues/475) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

</details>

---------------------------------------------------------------------------------------------------------

Thanks to:
https://roboflow.ai

https://ultralytics.com/

https://github.com/ultralytics/yolov5
