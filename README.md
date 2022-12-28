[![Colab Image Segmentation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/Segmentation_Depth_MLP.ipynb) >> Colab Image Segmentation


[![Colab MLP Train and Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/MLP_Image_Train_Inference_JAX.ipynb) >> Colab MLP Train and Inference


# Media-Segment-Depth-MLP
drafting Segmentation from mediapipe holistic estimating Depth using BoostingMonocularDepth with an added MLP Model in JAX

```
with mp_pose.Pose(static_image_mode=True, 
                          min_detection_confidence=0.2,
                          model_complexity=2, 
                          enable_segmentation=True,) as pose:
```
```
https://github.com/1kaiser/Media-Segment-Depth-MLP/blob/main/media/OUTPUT_FILE.mp4
```
https://user-images.githubusercontent.com/26379748/207690144-ef9530ac-62a9-459d-8c81-3c513b61c6da.mp4



```
with mp_pose.Pose(static_image_mode=False, 
                          min_detection_confidence=0.2,
                          model_complexity=2, 
                          enable_segmentation=True,) as pose:
```
```
https://github.com/1kaiser/Media-Segment-Depth-MLP/blob/main/media/OUTPUT_FILE%20(2).mp4
```
https://user-images.githubusercontent.com/26379748/207696727-b5da63f0-b6ab-46b0-93b1-e0a65698729c.mp4

