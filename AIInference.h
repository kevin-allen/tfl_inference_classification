#ifndef AIINFERENCE_H
#define AIINFERENCE_H

#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "opencv2/opencv.hpp"
/*
    * AIInference.h
    * Created on: 2024.04.12.
    * Class to perform inference using TF Lite models
    Typical usage:

    You would normally use the child class AIInferenceClassification
*/

class AIInference{
    /*Parent class that can be use for classification or 
    for running base convolutional models (without classification head) 

    This was created to avoid code duplication between classes
    */
    protected:
        std::string model_file_;
        std::unique_ptr<tflite::FlatBufferModel> model_;
        std::unique_ptr<tflite::Interpreter> interpreter_;
        TfLiteType input_tensor_type;
        size_t input_tensor_size_bytes;
        TfLiteType output_tensor_type;
        size_t output_tensor_size_bytes;
        int output_tensor_size;
        int image_width = 224;
        int image_height = 224;
        
        cv::Mat image_;
        void loadModel();
    public:
        AIInference();
        AIInference(std::string model_file);
        ~AIInference();
        
        void loadImage(std::string image_file);
        void preprocessImage();
        void normalizeImage();
        void copyImageToInputTensor();
        void runInference();
        void printInterpreterState();
};


#endif // AIINFERENCE_H