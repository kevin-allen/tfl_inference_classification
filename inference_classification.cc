/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// add openCV
#include  <filesystem>
#include "opencv2/opencv.hpp"


// to get indices of the top 5 classes
#include <algorithm>
#include <functional>
#include <queue>

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>
#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }
  


namespace fs = std::filesystem; // Create a namespace alias for convenience




// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void get_top_n(T* prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>>* top_results,
               TfLiteType input_type) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      top_result_pq;

  const long count = prediction_size;  // NOLINT(runtime/int)
  float value = 0.0;
  float max_value = 0.0;
  switch (input_type) {
      case kTfLiteFloat32:
        std::cout << "finding results in float32 output" << std::endl;
        break;
      case kTfLiteInt8:
        std::cout << "finding results in int8 output" << std::endl;
        break;
      case kTfLiteUInt8:
        std::cout << "finding results in uint8 output" << std::endl;
        break;
      default:
        std::cout << "unsupported output format" << std::endl;
        break;
    }
  
  std::cout << "prediction_size: " << prediction_size << std::endl;

  for (int i = 0; i < count; ++i) {
    switch (input_type) {
      case kTfLiteFloat32:
        value = prediction[i];
        break;
      case kTfLiteInt8:
        value = (prediction[i] + 128) / 256.0;
        break;
      case kTfLiteUInt8:
        value = prediction[i] / 255.0;
        break;
      default:
        break;
    }
    if (value > max_value) {
      max_value = value;
    }

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  std::cout << "max_value in prediction tensor: " << max_value << std::endl;

  
  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

















int main(int argc, char* argv[]) {

  int num_arg_needed=2; // required
  int non_opt_arguments; // given by the user
  const char* prog_name=argv[0];
  bool with_i_opt=false;

  int opt;
  while ((opt=getopt(argc, argv, "")) != -1)
    {
      switch (opt)
	    {
        default :
        {
          fprintf(stderr, "Usage: %s <tflite model> <image file>\n", prog_name);
          
          return 1;
        }
      }
    }

  non_opt_arguments=argc-optind; // number of non-option argument required
  if ((non_opt_arguments) !=  num_arg_needed){
    fprintf(stderr, "%s <tflite model> <image file>\n",prog_name);
    return 1;
  
  }

  const char* model_file = argv[optind];
  const char* image_file = argv[optind+1];

  std::cout << "Model: " << model_file << std::endl;
  std::cout << "Image: " << image_file << std::endl;

  // print openCV version
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_file);
  /*std::unique_ptr is smart pointer that owns and manages another object through a pointer
  and disposes of that object when the unique_ptr goes out of scope.
  */

  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  std::cout << "interpreter->tensors_size: " << interpreter->tensors_size() << std::endl;
  std::cout << "interpreter->nodes_size: " << interpreter->nodes_size() << std::endl;
  std::cout << "interpreter->inputs().size(): " << interpreter->inputs().size() << std::endl;
  std::cout << "interpreter->GetInputName(0) name: " << interpreter->GetInputName(0) << std::endl;
  std::cout << "interpreter->outputs().size(): " << interpreter->outputs().size() << std::endl;
  int input_type = interpreter->tensor(0)->type;
  std::cout << "input_type: " << input_type << std::endl;

  switch(input_type){
    case kTfLiteFloat32:
      std::cout << "input_type: kTfLiteFloat32" << std::endl;
      break;
    case kTfLiteUInt8:
      std::cout << "input_type: kTfLiteUInt8" << std::endl;
      break;
    case kTfLiteInt8:
      std::cout << "input_type: kTfLiteInt8" << std::endl;
      break;
    default:
      std::cout << "input_type: unknown" << std::endl;
      break;
  }
  

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());



  // check if the file mouse.png is present in the directory of the program 
  fs::path file_path{image_file};
  // Check if the file exists
  if (fs::exists(file_path)) {
      std::cout << "The image file " << file_path << " exists." << std::endl;
  } else {
      std::cout << "The file does not exist." << std::endl;
      return 1;
  }

  // open a png file with openCV
  cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
  if (image.empty())
  {
    std::cerr << "Could not read the image:" << file_path << std::endl;
    return 1;
  }
  // print the image dimensions
  std::cout << "Image dimensions: " << image.size() << std::endl;
  
  // resize teh image to 224x224
  cv::resize(image, image, cv::Size(224, 224));

  // print the image dimensions
  std::cout << "Image dimensions: " << image.size() << std::endl;
  
  // print the image dtype
  std::cout << "Image dtype: " << image.type() << std::endl;

  // transform the image to unsigned 8-bit integer
  switch(input_type){
    case kTfLiteFloat32:
      std::cout << "transform image to float" << std::endl;
      image.convertTo(image, CV_32F);
      break;
    case kTfLiteUInt8:
      std::cout << "transform image to unsigned int8" << std::endl;
      image.convertTo(image, CV_8U);
      break;
    case kTfLiteInt8:
      std::cout << "transform image to int8" << std::endl;
      image.convertTo(image, CV_8S);
      break;
    default:
      std::cout << "input_type: unknown" << std::endl;
      break;
  }
  
  // print the image dtype
  std::cout << "OpenCV image dtype: " << image.type() << std::endl;
  double min, max;
  cv::minMaxLoc(image, &min, &max);
  std::cout << "Image min value: " << min << std::endl;
  std::cout << "Image max value: " << max << std::endl;

  //copy the image to the input tensor
  int num_pixels = 224*224*3;

  // would be a good place to use a template function.
  switch(input_type){
    case kTfLiteFloat32:
    {
      std::cout << "copy float image to tensor" << std::endl;
      float* input = interpreter->typed_input_tensor<float>(0);
      // put our image in the model input
      for (int i = 0; i < num_pixels; ++i) {
          input[i] = image.at<float>(i);
        }
    }
      break;
    
    case kTfLiteUInt8:
    {
      std::cout << "copy uint8 image to tensor" << std::endl;
      uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
      // put our image in the model input
      for (int i = 0; i < num_pixels; ++i) {
          input[i] = image.at<uint8_t>(i);
        }
    }
      break;
    case kTfLiteInt8:
    {
      std::cout << "copy int8 image to tensor" << std::endl;
      int8_t* input = interpreter->typed_input_tensor<int8_t>(0);
      // put our image in the model input
      for (int i = 0; i < num_pixels; ++i) {
          input[i] = image.at<int8_t>(i);
        }
    }
      break;
    default:
      std::cout << "input_type: unknown" << std::endl;
      break;
  }  

 
  // Run inference
  std::cout << "Running inference ...\n";
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  //printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`




  int output = interpreter->outputs()[0];
  TfLiteType output_type = interpreter->tensor(output)->type;
  int nResults = 5;
  const float threshold = 0.001f;


  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;

  std::cout << "output_dims->size: " << output_dims->size << std::endl;

  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];

  std::cout << "output_size: " << output_size << std::endl;



  std::vector<std::pair<float, int>> top_results;

  std::cout << "calculating top results ..." << std::endl;
  switch (output_type) {
    case kTfLiteFloat32:
    {
      get_top_n<float>((float*) interpreter->typed_output_tensor<float>(0), 
                        output_size,nResults, threshold, &top_results,output_type);
    }
      break;
    case kTfLiteInt8:
      get_top_n<int8_t>(interpreter->typed_output_tensor<int8_t>(0),
                        output_size, nResults, threshold,
                        &top_results, output_type);
      break;
    case kTfLiteUInt8:
      get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                         output_size, nResults, threshold,
                         &top_results, output_type);
      break;
    default:
      std::cout << "cannot handle output type " << interpreter->tensor(0)->type << " yet" << std::endl;
      exit(-1);
  }







  std::cout << "Top results above threshold:" << std::endl;
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    std::cout << "index: " << index << ", prob: " << confidence << std::endl;
  }

  
  return 0;
}
