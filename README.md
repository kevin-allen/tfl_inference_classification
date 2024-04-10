# tfl_inference_classification

This provides code performing inference and GradCam using the TensorFlow Lite C++ API. 

Most of the example programs take a model and image as arguments.

## TF Lite Models

You should prepare your models in Python, using TensorFlow. Once they work well, convert them to TensorFlow Lite.

Examples of TensorFlow to TensorFlow Lite conversion: 

## AIInference Class

Perform inference using TF Lite models.

The models can be float32, uint8, or int8. 

The code is loosely based on [examples in the TensorFlow source repository](https://github.com/nxp-imx/tensorflow-imx/tree/lf-6.1.36_2.1.0/tensorflow/lite/examples/label_image).

## Compile these programs

You will need to have the source code for TensorFlow somewhere on your computer for compilation. For information: https://github.com/kevin-allen/phyBOARD/blob/main/tfl.md

### Clone the tfl_inference_classification repository
```
cd ~/repo
git clone https://github.com/kevin-allen/tfl_inference_classification.git
```

### Adapt the build instructions to find the location of the tensorflow repository.

Edit the CMakefile.txt to set the path to the tensorflow repository.

```
emacs CMakefile.txt
```

### Compile

```
cd ~/repo/tfl_inference_classification
mkdir build
cd build
cmake ..
make
```

The first compilation will take a while because CMake will recompile some of TensorFlow. The next time will be faster.


### Main programs

1. classifiy_image
2. 
