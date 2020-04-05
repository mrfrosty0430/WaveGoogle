# WaveGoogle

A gesture powered smart home device for accessibility using OpenPose


## Installation
1. Clone OpenPose from https://github.com/CMU-Perceptual-Computing-Lab/openpose and follow their installation guide
2. Clone this repository

## Building
1. run
``` /build/examples/openpose/openpose.bin --image_dir samples/ --write_images results --hand --write_json output ```
2. run
```python3 parse.py```
3. run
```python3 normalize.py```
4. run
```python3 nn.py``` OR ```python3 svm.py ```
