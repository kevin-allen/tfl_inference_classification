#include <cstdio>

// add openCV
#include "opencv2/opencv.hpp"

// filesystem tools
#include  <filesystem>

// include AIIinference.h
#include "AIInference.h"

// to get indices of the top 5 classes
#include <algorithm>
#include <functional>
#include <queue>


/*
./inference_classification /tmp/mobilenet_v1_1.0_224.tflite /home/kevin/repo/tensorflow/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp
*/
  

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
namespace fs = std::filesystem; // Create a namespace alias for convenience






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


  // create an instance of AIInference
  AIInference ai_inference(model_file);
  ai_inference.loadImage(image_file);
  ai_inference.preprocessImage();
  ai_inference.copyImageToInputTensor();
  ai_inference.runInference();
  ai_inference.copyResultTensorToResultArray();
  ai_inference.get_top_results();
  //ai_inference.printResults();


  /*
  
  




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

  */


  return 0;
}
