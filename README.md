# Image-Depth-Prediction

## Brief Introduction
We apply a ResNet to predict the object distance from the camera, based on the input video flow. Then accroding to this depth prediction, control signals are generated.

It can be used in robot obstacle avoidance.

## Result
(running on laptop with poor processing ability, low frame rate)

Demo


![image](/gif/Prediction.gif)

## Models

Download: [TensorFlow(.npy)](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy)

## Reference

[Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373)

Github： https://github.com/iro-cp/FCRN-DepthPrediction
