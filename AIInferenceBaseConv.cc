#include "AIInference.h"
#include "AIInferenceBaseConv.h"

#include  <filesystem>
#include <iostream>
#include <algorithm>
#include <functional>
#include <queue>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "opencv2/opencv.hpp"

// Write the AIInferenceClassification constructor
AIInferenceBaseConv::AIInferenceBaseConv(std::string model_file){
    std::cout << "AIInferenceBaseConv constructor" << std::endl;
    model_file_ = model_file;
    loadModel();
    allocateMemoryForResultArray();
}

AIInferenceBaseConv::~AIInferenceBaseConv(){
    // I wanted to get the results back to a float array, irrespective of the output tensor type.
    // Having a constant type simplifies the next steps.
    if (resultArray != nullptr) {
        std::cout << "Freeing memory for the result array" << std::endl;
        delete[] resultArray;
    }
}

void AIInferenceBaseConv::allocateMemoryForResultArray(){

    std::cout << "AIInferenceBaseConv::allocateMemoryForResultArray" << std::endl;
    
    // calculate the size of the output of the model
    int output =  interpreter_->outputs()[0];
    output_dims = interpreter_->tensor(output)->dims;    
    // print the dimensions of the output tensor
    for (int i = 0; i < output_dims->size; i++) {
        std::cout << "output_dims->data[" << i << "] = " << output_dims->data[i] << std::endl;
    }
    // calculate the size of the output tensor from output_dims
    output_tensor_size = 1;
    for (int i = 0; i < output_dims->size; i++) {
        output_tensor_size *= output_dims->data[i];
    }
    std::cout << "output_tensor_size = " << output_tensor_size << std::endl;

    // allocate memory for the result array (output of the network)
    resultArray = new float[output_tensor_size];
}

void AIInferenceBaseConv::copyResultTensorToResultArray(){
    std::cout << "AIInferenceBaseConv::copyResultTensorToResultArray" << std::endl;
    // Get the output tensor
    int output =  interpreter_->outputs()[0];
    

    switch(output_tensor_type){
        case kTfLiteFloat32:
            {
            std::cout << "Output_type: kTfLiteFloat32" << std::endl;
            // Copy the output tensor to the result array
            std::memcpy(resultArray, interpreter_->typed_output_tensor<float>(0), output_tensor_size*sizeof(float));
            }
            break;
        case kTfLiteInt8:
            {
                std::cout << "Output_type: kTfLiteInt8" << std::endl;
                int8_t* output = interpreter_->typed_output_tensor<int8_t>(0);
                for(int i = 0; i < output_tensor_size; i++)
                    resultArray[i] = (output[i] + 128)/256.0;
            }
            break;
        case kTfLiteUInt8:
            {
                std::cout << "Output_type: kTfLiteUInt8" << std::endl;
                uint8_t* output = interpreter_->typed_output_tensor<uint8_t>(0);
                for(int i = 0; i < output_tensor_size; i++)
                    resultArray[i] = output[i]/256.0;
            }
            break;
        default:
            std::cerr << "Output_type: unknown" << std::endl;
            exit(1);
    }

    // Print the maximum value in the resultsArray
    float max_value = *std::max_element(resultArray, resultArray + output_tensor_size);
    std::cout << "max_value in resultArray: " << max_value << std::endl;
    // Print the index of the maximal value
    int max_index = std::distance(resultArray, std::max_element(resultArray, resultArray + output_tensor_size));
    std::cout << "index of max_value in resultArray: " << max_index << std::endl;
}

float* AIInferenceBaseConv::getResultArray(){
    return resultArray;
}