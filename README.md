<table>
<thead>
<tr>
<td>


<a href="https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/nnx_string3d_experiment" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> >> JAX FLAX nnx STRING_3D performance test using ViTTiny 6 layers on TPUv5e1

<table>
  <tr>
    <td>
      <img src="https://github.com/1kaiser/Media-Segment-Depth-MLP/blob/main/media/893c65c6-772a-495f-9e90-7e4de61b88f5.png" width="300"/>
    </td>
    <td>
      <img src="https://github.com/1kaiser/Media-Segment-Depth-MLP/blob/main/media/752baa62-3794-4b1f-8b88-6c86cef966f0.png" width="300"/>
    </td>
    <td>
      <img src="https://github.com/1kaiser/Media-Segment-Depth-MLP/blob/main/media/71971b7a-e3cc-433b-a9d3-500b3a375161.png" width="300"/>
    </td>
  </tr>
</table>

<a href="https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/STRING_3D_ViTTiny_Elayers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> >> JAX STRING_3D performance test using ViTTiny 4 to 12 layers on TPUv8

<table>
  <tr>
    <td>
      <img src="https://github.com/1kaiser/Media-Segment-Depth-MLP/blob/main/media/62a4fb0e-8d44-4bdd-8df9-82a9d4e5c2d0.png" width="300"/>
    </td>
    <td>
      <img src="https://github.com/1kaiser/Media-Segment-Depth-MLP/blob/main/media/80c7e2a0-8afc-44e3-81f7-b80e56ed53ef.png" width="300"/>
    </td>
  </tr>
</table>



<a href="https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/2D_Positional_Encoding_Vision_Transformer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> >> JAX 2D_Positional_Encoding_Vision_Transformer

[![Colab MLP Train and Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/MLP_Image_JAX.ipynb) >> Colab MLP Image Train and Inference Serial


[![Colab MLP Train and Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/MLP_Image_training_Parallel.ipynb) >> Colab MLP Train and Inference Parallel



[![Colab MLP Train and Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/MLP_Image_Train_Inference_JAX.ipynb) >> Colab MLP Train and Inference


[![Colab MLP Train and Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/EfficientSAM_example.ipynb) >> Colab **[EfficientSAM_example](https://github.com/yformer/EfficientSAM)**


</td>
</tr>
</tbody>
</table>


<table>
<thead>
<tr>
<td>




![Network diagram](https://user-images.githubusercontent.com/3310961/85066930-ad444580-b164-11ea-9cc0-17494679e71f.png)


</td>
</tr>
</tbody>
</table>


<table>
<thead>
<tr>
<td>





[![Colab Image Segmentation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Media-Segment-Depth-MLP/blob/main/Segmentation_Depth_MLP.ipynb) >> Colab Image Segmentation




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

MediaPipe + EfficientSAM

https://github.com/1kaiser/Media-Segment-Depth-MLP/assets/26379748/452827a0-0655-436a-a989-60341c2aee1f.mp4




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




</td>
</tr>
</tbody>
</table>
