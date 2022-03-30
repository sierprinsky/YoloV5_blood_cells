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

## <div align="center">Thanks to:</div>

https://roboflow.ai

https://ultralytics.com/

https://github.com/ultralytics/yolov5

---------------------------------------------------------------------------------------------------------

## <div align="center">Training results summary</div>


<details>
<summary>YOLOv5, nano model:</summary>

```bash
50 epochs completed in 1.421 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 3.8MB
Optimizer stripped from runs/train/exp/weights/best.pt, 3.8MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model summary: 213 layers, 1763224 parameters, 0 gradients, 4.2 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:  12%|█▎        | 1/8 [00:02<00:18,  2.71s/it]                                                         WARNING: NMS time limit 0.300s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:  25%|██▌       | 2/8 [00:08<00:26,  4.41s/it]                                                         WARNING: NMS time limit 0.300s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:  38%|███▊      | 3/8 [00:14<00:25,  5.04s/it]                                                         WARNING: NMS time limit 0.300s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 8/8 [00:29<00:00,  3.65s/it]                                                         
                 all         73        967      0.847      0.644      0.669      0.439
           Platelets         73         76      0.842      0.632      0.664      0.336
                 RBC         73        819      0.742      0.565      0.605      0.407
                 WBC         73         72      0.957      0.736      0.736      0.572

```
</details>
 

<details>
<summary>YOLOv5, small model:</summary>

```bash
50 epochs completed in 3.031 hours.
Optimizer stripped from runs/train/exp2/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/exp2/weights/best.pt, 14.3MB

Validating runs/train/exp2/weights/best.pt...
Fusing layers... 
Model summary: 213 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:  12%|█▎        | 1/8 [00:02<00:20,  2.91s/it]                                                         WARNING: NMS time limit 0.300s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:  38%|███▊      | 3/8 [00:16<00:30,  6.00s/it]                                                         WARNING: NMS time limit 0.300s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 8/8 [00:31<00:00,  3.98s/it]                                                         
                 all         73        967      0.877      0.824      0.854      0.587
           Platelets         73         76      0.834      0.816       0.83       0.45
                 RBC         73        819       0.84       0.74      0.831      0.585
                 WBC         73         72      0.956      0.917      0.902      0.726

```
</details>


<details>
<summary>YOLOv5, medium model:</summary>

```bash
50 epochs completed in 6.814 hours.
Optimizer stripped from runs/train/exp3/weights/last.pt, 42.1MB
Optimizer stripped from runs/train/exp3/weights/best.pt, 42.1MB

Validating runs/train/exp3/weights/best.pt...
Fusing layers... 
Model summary: 290 layers, 20861016 parameters, 0 gradients, 48.0 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:  38%|███▊      | 3/8 [00:25<00:43,  8.78s/it]                                                         WARNING: NMS time limit 0.300s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 8/8 [00:55<00:00,  6.95s/it]                                                         
                 all         73        967      0.905      0.857      0.908       0.63
           Platelets         73         76      0.875      0.829      0.904      0.494
                 RBC         73        819       0.87      0.744      0.837      0.592
                 WBC         73         72      0.969          1      0.983      0.804

```
</details>


<details>
<summary>YOLOv5, large model:</summary>

```bash
50 epochs completed in 9.953 hours.
Optimizer stripped from runs/train/exp4/weights/last.pt, 92.8MB
Optimizer stripped from runs/train/exp4/weights/best.pt, 92.8MB

Validating runs/train/exp4/weights/best.pt...
Fusing layers... 
Model summary: 367 layers, 46119048 parameters, 0 gradients, 107.8 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 8/8 [00:39<00:00,  4.93s/it]                                                         
                 all         73        967      0.845      0.913      0.913      0.637
           Platelets         73         76      0.817      0.882      0.875      0.509
                 RBC         73        819      0.758      0.856      0.879      0.624
                 WBC         73         72      0.961          1      0.986      0.778
Results saved to runs/train/exp4
```
</details>

 
 
 
 
 
 ## <div align="center">Validation results summary</div>


<details>
<summary>YOLOv5, nano model:</summary>

```bash
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 3/3 [00:20<00:00,  6.99s/it]                                                         
                 all         73        967      0.835      0.598      0.608      0.411
           Platelets         73         76      0.789      0.592      0.587      0.293
                 RBC         73        819      0.753      0.551      0.585      0.414
                 WBC         73         72      0.962      0.653      0.651      0.524
Speed: 3.5ms pre-process, 146.8ms inference, 22.1ms NMS per image at shape (32, 3, 416, 416)
Results saved to runs/val/exp2

```
</details>
 

<details>
<summary>YOLOv5, small model:</summary>

```bash
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 3/3 [00:27<00:00,  9.28s/it]                                                         
                 all         73        967      0.864      0.831      0.856      0.594
           Platelets         73         76      0.824      0.801      0.803      0.432
                 RBC         73        819      0.819      0.761      0.844      0.595
                 WBC         73         72      0.948      0.931      0.921      0.754
Speed: 3.9ms pre-process, 289.1ms inference, 16.4ms NMS per image at shape (32, 3, 416, 416)
Results saved to runs/val/exp

```
</details>


<details>
<summary>YOLOv5, medium model:</summary>

```bash
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 3/3 [00:45<00:00, 15.23s/it]                                                         
                 all         73        967      0.871      0.893      0.911      0.636
           Platelets         73         76      0.826      0.877      0.891      0.492
                 RBC         73        819      0.822      0.801       0.86       0.61
                 WBC         73         72      0.966          1      0.983      0.805
Speed: 4.5ms pre-process, 562.6ms inference, 4.7ms NMS per image at shape (32, 3, 416, 416)
Results saved to runs/val/exp3

```
</details>


<details>
<summary>YOLOv5, large model:</summary>

```bash
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 3/3 [01:19<00:00, 26.44s/it]                                                         
                 all         73        967       0.85      0.911      0.914      0.639
           Platelets         73         76      0.817      0.882      0.876       0.51
                 RBC         73        819      0.771      0.851      0.882       0.63
                 WBC         73         72      0.961          1      0.986      0.778
Speed: 4.0ms pre-process, 1049.5ms inference, 3.1ms NMS per image at shape (32, 3, 416, 416)
Results saved to runs/val/exp4

```
</details>
