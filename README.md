# tfl_inference_classification

This provides code performing inference and GradCam using the TensorFlow Lite C++ API. 

Most of the example programs take a model and image as arguments.

# To do

Organize the code so the inference code is re-usable (separate class).


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

If you don't have the tensorflow source repository on your computer, clone it from GitHub.

```
git clone https://github.com/tensorflow/tensorflow.git
```

### Adapt the build instructions to find the location of the tensorflow repository.

Edit the CMakefile.txt to set the path to the tensorflow repository.

In the repo, this points to `/home/kevin/repo/tensorflow`. Just change it to the location of the TensorFlow repository on your system.

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

### Test a program

Download a TensorFlow Lite model from Google.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp
```

Perform inference with the program `inference_classification`

```
cd ~/repo/tfl_inference_classification/build
./inference_classification /tmp/mobilenet_v1_1.0_224.tflite /home/kevin/repo/tensorflow/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp 
```


### Main programs

1. classifiy_image
2. 
