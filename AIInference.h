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
*/

class AIInference{
    private:
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
        int nResults = 5;
        std::vector<std::pair<float, int>> top_results;
        float threshold = 0.001;
        float* resultArray = nullptr; // array to store the output of the model
        cv::Mat image_;
        void loadModel();
        void allocate_memory_for_result_array();

       
  
        
        void postprocess();
    public:
        AIInference(std::string model_file);
        ~AIInference();
        void loadImage(std::string image_file);
        void preprocessImage();
        void copyImageToInputTensor();
        void runInference();
        void printResults();
        void copyResultTensorToResultArray();
        void get_top_results();
        
        void printInterpreterState();
        ;
    
};




#endif // AIINFERENCE_H