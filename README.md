# Depth-perception-with-laser

We are able to detect depth with the following system
<p align="">
  <img src="https://github.com/AlbertoMoca/Depth-perception-with-laser/blob/main/images_readme/system.png" width="400" />
</p>

For this we need to first solve the deformation caused by the camera lens in most cases this is radial so we can make an approximation with some marks and changing depth of the pictures

<p align="">
  <img src="https://github.com/AlbertoMoca/Depth-perception-with-laser/blob/main/images_readme/camera_deformation_calibration.png"  width="400"/>
</p>

Then we can do the approximation of the depth changing the camera position and measuring the distance to the center and making an approximation of the surface

<p align="">
  <img src="https://github.com/AlbertoMoca/Depth-perception-with-laser/blob/main/images_readme/depth_calibration.png"  width="400"/>
</p>
<p align="">
  <img src="https://github.com/AlbertoMoca/Depth-perception-with-laser/blob/main/images_readme/aproximation_result.png"  width="400"/>
</p>

### Sample result

<p align="">
  <img src="https://github.com/AlbertoMoca/Depth-perception-with-laser/blob/main/images_readme/result.png"  width="400"/>
</p>

|     | depth | width | height |  
| --- | --- | --- | --- |
| real | 64.0cm | 30.0cm | 23.0cm |
| prediction | 62.6cm | 30.1cm | 26.2cm |

