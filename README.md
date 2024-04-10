# tfl_inference_classification

This provides code to perform inference using the TensorFlow Lite C++ API. 

It contains a AIInference class that you can use to perform inference using TF Lite models.

The code is based on some examples provided with the [TensorFlow source code](https://github.com/nxp-imx/tensorflow-imx/tree/lf-6.1.36_2.1.0/tensorflow/lite/examples/label_image). I took some inspiration from the label_image example.

## Start developing 

You will need to have the source code for TensorFlow and you should be able to compile it on your machine. For information: https://github.com/kevin-allen/phyBOARD/blob/main/tfl.md


```
cd ~/repo
git clone https://github.com/kevin-allen/tfl_inference_classification.git
git clone 
cd ~/repo/tensorflow/tensorflow/lite/examples/
ln -s ~/repo/tfl_inference_classification/ . # will not compile if it is a symbolic link!!!!!
```


## Compile

```
cd ~/repo/tensorflow/tensorflow/lite/examples/tfl_inference_classification
mkdir build
cd build
cmake ..
make
```

The first compilation will take a while because CMake will recompile TensorFlow
