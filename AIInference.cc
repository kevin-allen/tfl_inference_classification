#include "AIInference.h"

#include  <filesystem>
#include <iostream>
#include <algorithm>
#include <functional>
#include <queue>

#include "opencv2/opencv.hpp"

// Write the AIInference initializer
AIInference::AIInference(std::string model_file){
    model_file_ = model_file;
    loadModel();
    
}

AIInference::~AIInference(){
    std::cout << "Destroy AIInference object" << std::endl;
    if (resultArray != nullptr)
        delete[] resultArray;
}

void AIInference::allocate_memory_for_result_array(){
    // I wanted to get the results back to a float array, irrespective of the output tensor type.
    // Having a constant type simplifies the next steps.
    if (resultArray != nullptr) {
        std::cout << "Freeing memory for the result array" << std::endl;
        delete[] resultArray;
    }

    // calculate the size of the output of the model
    int output =  interpreter_->outputs()[0];
    TfLiteIntArray* output_dims = interpreter_->tensor(output)->dims;    
    // assume output dims to be something like (1, 1, ... ,size)
    output_tensor_size = output_dims->data[output_dims->size - 1]; // 

    // allocate memory for the result array (output of the network)
    resultArray = new float[output_tensor_size];
}

// Write the loadModel function
void AIInference::loadModel(){
    // Check if the model file exists
    if (!std::filesystem::exists(model_file_)) {
        std::cerr << "Model file " << model_file_ << " does not exist" << std::endl;
        exit(1);
    }
    
    // Load the TFLite model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_file_.c_str());
    if (model_ == nullptr) {
        std::cerr << "Failed to load the model" << std::endl;
        exit(1);
    }

    // Create the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    if (interpreter_==nullptr) {
        std::cerr << "Failed to create the interpreter" << std::endl;
        exit(1);
    }

    // Allocate tensor buffers
    interpreter_->AllocateTensors();

    // Get input tensor details
    input_tensor_type = interpreter_->tensor(0)->type;
    input_tensor_size_bytes = interpreter_->tensor(0)->bytes;

    // Print input tensor details
    switch(input_tensor_type){
        case kTfLiteFloat32:
        std::cout << "Input_type: kTfLiteFloat32" << std::endl;
        break;
        case kTfLiteUInt8:
        std::cout << "Input_type: kTfLiteUInt8" << std::endl;
        break;
        case kTfLiteInt8:
        std::cout << "Input_type: kTfLiteInt8" << std::endl;
        break;
        default:
        std::cerr << "Input_type: unknown" << std::endl;
        exit(1);
        break;
    }  
    std::cout << "Input tensor size in bytes: " << input_tensor_size_bytes << std::endl;

    // Get output tensor details
    int output =  interpreter_->outputs()[0];
    output_tensor_type = interpreter_->tensor(output)->type;
    output_tensor_size_bytes = interpreter_->tensor(output)->bytes;


    // Print output tensor details
    switch(output_tensor_type){
        case kTfLiteFloat32:
        std::cout << "Output_type: kTfLiteFloat32" << std::endl;
        break;
        case kTfLiteUInt8:
        std::cout << "Output_type: kTfLiteUInt8" << std::endl;
        break;
        case kTfLiteInt8:
        std::cout << "Output_type: kTfLiteInt8" << std::endl;
        break;
        default:
        std::cerr << "Output_type: unknown" << std::endl;
        exit(1);
        break;
    }
    std::cout << "Output tensor size in bytes: " << output_tensor_size_bytes << std::endl;

    allocate_memory_for_result_array();
}


// Write the loadInput function
void AIInference::loadImage(std::string image_file){
    
    std::cout << "Load image " << image_file << std::endl;
    // Check if the image file exists
    if (!std::filesystem::exists(image_file)) {
        std::cerr << "Image file " << image_file << " does not exist" << std::endl;
        exit(1);
    }
    // Load the image file
    image_ = cv::imread(image_file, cv::IMREAD_COLOR);
    if (image_.empty()) {
        std::cerr << "Failed to load the image" << std::endl;
        exit(1);
    }

}

void AIInference::preprocessImage(){
    
    // check that we have a valid image
    if (image_.empty()) {
        std::cerr << "Image is empty, run AIInference::loadImage() before calling AIInference::preprocessImage()" << std::endl;
        exit(1);
    }
    

    cv::Scalar mean = cv::mean(image_);
    std::cout << "Mean values from cv::mean(image_): " << mean << std::endl;


    // Resize the image
    std::cout << "Resize image to " << image_width << "x" << image_height << std::endl;
    cv::resize(image_, image_, cv::Size(image_width, image_height));

  
    // Convert the image to RGB
    std::cout << "Convert image to RGB" << std::endl;
    cv::cvtColor(image_, image_, cv::COLOR_BGR2RGB);


    // Convert the image to the right type
    switch(input_tensor_type){
        case kTfLiteFloat32:
        std::cout << "Convert image to float" << std::endl;
        image_.convertTo(image_, CV_32F);
        break;
        case kTfLiteUInt8:
        std::cout << "Convert image to unsigned int8" << std::endl;
        image_.convertTo(image_, CV_8U);
        break;
        case kTfLiteInt8:
        std::cout << "Convert image to int8" << std::endl;
        image_.convertTo(image_, CV_8S);
        break;
        default:
        std::cerr << "Input_type: unknown" << std::endl;
        exit(1);
    }
    
    
    // normalize the image
    // this will depend on the model and dataset used to train it
    switch(input_tensor_type){
        case kTfLiteFloat32:
        {
            std::cout << "Normalize float image (x-127.5)/127.5" << std::endl;
            float input_mean = 127.5f; // to compare to label_image.cc
            float input_std = 127.5f;
            for(int i=0; i < 224*224*3;i++){
                image_.at<float>(i) = (image_.at<float>(i) - input_mean)/input_std;
            }

        }
        break;
        case kTfLiteUInt8:
        {
        std::cout << "!! No normalization of uint8 image" << std::endl;
        }
        break;
        case kTfLiteInt8:
        {
        std::cout << "!! No normalization of int8 image" << std::endl;
        }
        break;
        default:
        std::cerr << "!! Input_type: unknown" << std::endl;
        exit(1);
    }


    std::cout << "First pixel value: " << image_.at<cv::Vec3f>(0, 0) << std::endl;
    // calculate the mean value for red, green and blue in the image
    mean = cv::mean(image_);
    std::cout << "Mean values from cv::mean(image_): " << mean << std::endl;
    std::cout << "depth: " << image_.depth() << std::endl;
    std::cout << "channels: " << image_.channels() << std::endl;
    std::cout << "type: " << image_.type() << std::endl;
    std::cout << "size: " << image_.size() << std::endl;


}

void AIInference::copyImageToInputTensor(){
    // Copy the image to the input tensor

    int num_pixels = image_width*image_height*3;



    switch(input_tensor_type){
        case kTfLiteFloat32:
        {
        std::cout << "Copy image to input tensor of type float" << std::endl;
        memcpy(interpreter_->typed_input_tensor<float>(0), image_.ptr<float>(0), 
        image_width*image_height*3*sizeof(float));
        }
        break;
        case kTfLiteUInt8:
        {
        std::cout << "Copy image to input tensor of type unsigned int8" << std::endl;
        uint8_t* input = interpreter_->typed_input_tensor<uint8_t>(0);
        // put our image in the model input
        for (int i = 0; i < num_pixels; ++i) {
          input[i] = image_.at<uint8_t>(i);
        }
        }
        //memcpy(interpreter_->typed_input_tensor<uint8_t>(0), image_.data, input_tensor_size_bytes);
        break;
        case kTfLiteInt8:
        {
        std::cout << "Copy image to input tensor of type int8" << std::endl;
        int8_t* input = interpreter_->typed_input_tensor<int8_t>(0);
        // put our image in the model input
        for (int i = 0; i < num_pixels; ++i) {
          input[i] = image_.at<int8_t>(i);
        }
        }
        //memcpy(interpreter_->typed_input_tensor<int8_t>(0), image_.data, input_tensor_size_bytes);
        break;
        default:
        std::cerr << "Input_type: unknown" << std::endl;
        exit(1);
    }
}

// Write the runInference function
void AIInference::runInference(){
      
    // Run the inference
    std::cout << "Run inference" << std::endl;
    interpreter_->Invoke();
}

void AIInference::copyResultTensorToResultArray(){
    
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

    float max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < output_tensor_size; ++i) {
        if (resultArray[i] > max_value) {
            max_value = resultArray[i];
            max_index = i;
        }
    }
    std::cout << "Max value at index " << max_index << " in the result array: " << max_value << std::endl;

}


void AIInference::get_top_results()
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

    std::cout << "Top results above threshold:" << std::endl;
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        std::cout << "index: " << index << ", prob: " << confidence << std::endl;
    }

 }


void AIInference::printInterpreterState(){
    // Print the interpreter state
    tflite::PrintInterpreterState(interpreter_.get());
}
