# WaveGoogle

A gesture powered smart home device for accessibility using OpenPose


## Installation
1. Clone OpenPose from https://github.com/CMU-Perceptual-Computing-Lab/openpose and follow their installation guide
2. Clone this repository

## Building the Dataset
1. run
``` /build/examples/openpose/openpose.bin --image_dir samples/ --write_images results --hand --write_json output ```
2. run
```python3 parse.py```
3. run
```python3 condense.py```
4. run
```python3 normalize_v2.py```
5. run
```python3 svm_v2.py ```
This will save the SVM Classifier as a .pkl object, which you can load

## Testing the Classifier
1. Obtain the Classifier (.pkl file)
2. run
``` python3 predict.py```
