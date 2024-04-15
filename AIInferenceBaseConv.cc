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
    allocate_memory_for_result_array();
}

AIInferenceBaseConv::~AIInferenceBaseConv(){
    // I wanted to get the results back to a float array, irrespective of the output tensor type.
    // Having a constant type simplifies the next steps.
    if (resultArray != nullptr) {
        std::cout << "Freeing memory for the result array" << std::endl;
        delete[] resultArray;
    }
}

void AIInferenceBaseConv::allocate_memory_for_result_array(){

    std::cout << "AIInferenceBaseConv::allocate_memory_for_result_array" << std::endl;
    
    // calculate the size of the output of the model
    int output =  interpreter_->outputs()[0];
    TfLiteIntArray* output_dims = interpreter_->tensor(output)->dims;    
    // assume output dims to be something like (1, 1, ... ,size)
    output_tensor_size = output_dims->data[output_dims->size - 1]; // 
    // allocate memory for the result array (output of the network)
    resultArray = new float[output_tensor_size];
}