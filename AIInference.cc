#include "AIInference.h"

#include  <filesystem>
#include <iostream>
#include <algorithm>
#include <functional>
#include <queue>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "opencv2/opencv.hpp"

// Write the AIInference constructor
AIInference::AIInference(){
    std::cout << "AIInference constructor" << std::endl;
}

AIInference::~AIInference(){
   
}


// Write the loadModel function
void AIInference::loadModel(){
    std::cout << "Load model: " << model_file_ << std::endl;
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

    // Check how many inputs and outputs the model has
    std::cout << "Number of inputs: " << interpreter_->inputs().size() << std::endl;
    std::cout << "Number of outputs: " << interpreter_->outputs().size() << std::endl;

    // Get input tensor details
    int input =  interpreter_->inputs()[0];

    input_tensor_type = interpreter_->tensor(input)->type;
    input_tensor_size_bytes = interpreter_->tensor(input)->bytes;
    input_dims = interpreter_->tensor(input)->dims;
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

    // print the dimensions of the input tensor
    for (int i = 0; i < input_dims->size; i++) {
        std::cout << "input_dims->data[" << i << "] = " << input_dims->data[i] << std::endl;
    }
    std::cout << "Input tensor size in bytes (seems wrong): " << input_tensor_size_bytes << std::endl;

    // Get output tensor details
    int output =  interpreter_->outputs()[0];
    output_tensor_type = interpreter_->tensor(output)->type;
    output_tensor_size_bytes = interpreter_->tensor(output)->bytes;
    output_dims = interpreter_->tensor(output)->dims;    
    
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
    // print the dimensions of the output tensor
    for (int i = 0; i < output_dims->size; i++) {
        std::cout << "output_dims->data[" << i << "] = " << output_dims->data[i] << std::endl;
    }

    std::cout << "Output tensor size in bytes: " << output_tensor_size_bytes << std::endl;

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

    mean = cv::mean(image_);
    std::cout << "Mean values from cv::mean(image_): " << mean << std::endl;
}

void AIInference::normalizeImage(){
    /////////////////////////////////////////////////////////////////////////////////////////
    // Data normalization ///////////////////////////////////////////////////////////////////
    // This step will depend on the model and how inputs were processed during training.  ///
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "Normalize image" << std::endl;
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

    cv::Scalar mean = cv::mean(image_);
    std::cout << "Mean values from cv::mean(image_): " << mean << std::endl;
    //std::cout << "First pixel value: " << image_.at<cv::Vec3f>(0, 0) << std::endl;
    // calculate the mean value for red, green and blue in the image
    //cv::Scalar mean = cv::mean(image_);
    //std::cout << "Mean values from cv::mean(image_): " << mean << std::endl;
    //std::cout << "depth: " << image_.depth() << std::endl;
    //std::cout << "channels: " << image_.channels() << std::endl;
    //std::cout << "type: " << image_.type() << std::endl;
    //std::cout << "size: " << image_.size() << std::endl;

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

void AIInference::setInputTensorFromBaseConvResultArray(float* resultArray){
    std::cout << "AIInference::setImageFromBaseConvResultArray(float* resultArray)" << std::endl;

    // calculate the input tensor size from input_dims
    int size = 1;
    for (int i = 0; i < input_dims->size; i++) {
        size *= input_dims->data[i];
    }
    std::cout << "size = " << size << std::endl;

    switch(input_tensor_type){
        case kTfLiteFloat32:
        {
            std::cout << "Copy image to input tensor of type float" << std::endl;
            memcpy(interpreter_->typed_input_tensor<float>(0), resultArray, 
            size*sizeof(float));
        }
        break;
        case kTfLiteUInt8:
        {
        std::cout << "Copy image to input tensor of type unsigned int8" << std::endl;
        std::cout << "Not implemented yet" << std::endl;
        exit(1);
        }
        //memcpy(interpreter_->typed_input_tensor<uint8_t>(0), image_.data, input_tensor_size_bytes);
        break;
        case kTfLiteInt8:
        {
        std::cout << "Copy image to input tensor of type int8" << std::endl;
        std::cout << "Not implemented yet" << std::endl;
        exit(1);
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
    auto start = std::chrono::high_resolution_clock::now();
    interpreter_->Invoke();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Inference duration: " << duration.count()*1000 << " ms" << std::endl;

    //start = std::chrono::high_resolution_clock::now();
    //int repetitions = 10;
    //for(int i = 0; i < repetitions; i++){
     //   interpreter_->Invoke();
    //}
    //end = std::chrono::high_resolution_clock::now();
    //duration = end - start;
    //std::cout << "Average inference duration ("<<repetitions<<" rep): " << duration.count()*1000/repetitions << " ms" << std::endl;



}


void AIInference::printInterpreterState(){
    // Print the interpreter state
    tflite::PrintInterpreterState(interpreter_.get());
}
