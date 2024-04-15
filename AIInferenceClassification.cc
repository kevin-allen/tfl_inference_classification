#include "AIInference.h"
#include "AIInferenceClassification.h"

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
AIInferenceClassification::AIInferenceClassification(std::string model_file){
    std::cout << "AIInferenceClassification constructor" << std::endl;
    model_file_ = model_file;
    loadModel();
    allocateMemoryForResultArray();
}

AIInferenceClassification::~AIInferenceClassification(){
    // I wanted to get the results back to a float array, irrespective of the output tensor type.
    // Having a constant type simplifies the next steps.
    if (resultArray != nullptr) {
        std::cout << "Freeing memory for the result array" << std::endl;
        delete[] resultArray;
    }
}

void AIInferenceClassification::allocateMemoryForResultArray(){
    
    // calculate the size of the output of the model
    int output =  interpreter_->outputs()[0];
    TfLiteIntArray* output_dims = interpreter_->tensor(output)->dims;    
    // assume output dims to be something like (1, 1, ... ,size)
    output_tensor_size = output_dims->data[output_dims->size - 1]; // 
    // allocate memory for the result array (output of the network)
    resultArray = new float[output_tensor_size];
}

void AIInferenceClassification::setLabels(std::string label_file){
    // Check if the label file exists
    label_file_= label_file;
    if (!std::filesystem::exists(label_file_)) {
        std::cerr << "Label file " << label_file_ << " does not exist" << std::endl;
        exit(1);
    }
    // Load the label file
    std::ifstream label_file_stream(label_file_);
    if (!label_file_stream.is_open()) {
        std::cerr << "Failed to open label file " << label_file_ << std::endl;
        exit(1);
    }
    // Read the labels
    std::string line;
    while (std::getline(label_file_stream, line)) {
        labels.push_back(line);
    }
    label_file_stream.close();
    std::cout << "Loaded " << labels.size() << " labels" << std::endl;
}



void AIInferenceClassification::copyResultTensorToResultArray(){
    
    std::cout << "Copy the result tensor to the result array" << std::endl;
    switch (output_tensor_type) {
            case kTfLiteFloat32:
            {
                memcpy(resultArray, interpreter_->typed_output_tensor<float>(0), output_tensor_size*sizeof(float));
            }
            break;
            case kTfLiteInt8:
            {
                int8_t* output = interpreter_->typed_output_tensor<int8_t>(0);
                for(int i = 0; i < output_tensor_size; i++)
                    resultArray[i] = (output[i] + 128)/256.0;
            }
            break;
            case kTfLiteUInt8:
            {
                uint8_t* output = interpreter_->typed_output_tensor<uint8_t>(0);
                for(int i = 0; i < output_tensor_size; i++)
                    resultArray[i] = output[i]/256.0;
            }
            break;
            default:
            std::cerr << "cannot handle output type " << interpreter_->tensor(0)->type << " yet" << std::endl;
            exit(-1);
        }
}


void AIInferenceClassification::getTopResults()
 {
    // Get the top N results and store in top_results
    
  // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      top_result_pq;

    //  empty the top_results vector
    top_results.clear();
    
    float max_value = 0.0;
    for (int i = 0; i < output_tensor_size; ++i) {
            
    if (resultArray[i] > max_value) {
      max_value = resultArray[i];
    }

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (resultArray[i] < threshold) {
      continue;
    }
    top_result_pq.push(std::pair<float, int>(resultArray[i], i));
    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > nResults) {
      top_result_pq.pop();
        }
    }
  
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results.push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results.begin(), top_results.end());
 }

void AIInferenceClassification::printTopResults(){
    // Print the top results
    std::cout << std::fixed << std::setprecision(5);
    if (labels.size() == 0) {
        std::cout << "No labels loaded" << std::endl;
        for (const auto& result : top_results) {
            const float confidence = result.first;
            const int index = result.second;
            std::cout << "index: " << index << ", prob: " << confidence << std::endl;
        }
        return;
    } else {

        if (output_tensor_size != labels.size()) {
            std::cerr << "Output tensor size does not match the number of labels" << std::endl;
            exit(1);
        }
        for (const auto& result : top_results) {
            const float confidence = result.first;
            const int index = result.second;
            std::cout << "index: " << index << ", prob: " << confidence << ", label: " << labels[index] << std::endl;

        
        }
        return;
    }
}

