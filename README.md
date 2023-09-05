# Real-Time-Digit-Recognition-Using-CNN
Train a convolutional neural network on the MNIST dataset to recognize handwritten digits and use the trained model for real-time recognition from a live video feed.

## Overview
`train.py` : Train the model

`detection.py` : Detect digits using live cam

`model.h5` : Sample model (pretrained)

## Prerequisites
|  Dependencies |    Version    |
| ------------- | ------------- |
| Python        | 3.7           |
| Keras         | 2.3.1         |
| Numpy         | 1.21.5        |
| OpenCV        | 4.8.0         |
| h5py          | 2.10.0        |

## Usage
Run 'train.py' to train the model:
```
python train.py
```
Once the model has been trained, run 'detection.py':
```
python detection.py
```
A sample model with an accuracy of 99.19% has been provided

## Demonstration


https://github.com/j16m3n4m6y4l/Real-Time-Digit-Recognition-Using-CNN/assets/132979609/1e8fb22e-21e2-44cc-bd66-a12929726e9c


## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

[Training](https://www.hackster.io/dhq/ai-digit-recognition-with-picamera-2c017f)

[Detection](https://github.com/dhanpalrajpurohit/handwritten-digit-detector)
