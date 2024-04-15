#include <cstdio>
#include "opencv2/opencv.hpp"
#include  <filesystem>
#include "AIInferenceBaseConv.h"
#include <string>

/*
./convolution_base_model /tmp/base_model.tflite ~/repo/tensorflow/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp 
See https://github.com/kevin-allen/tfl_inference_classification/blob/main/README.md for more information
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

  std::string model_file = argv[optind];
  std::string image_file = argv[optind+1];

  std::cout << "Model: " << model_file << std::endl;
  std::cout << "Image: " << image_file << std::endl;
  

  // create an instance of AIInference
  AIInferenceBaseConv ai_inference(model_file); // create an instance of AIInference and load the model into memory
  //ai_inference.loadImage(image_file);
  //ai_inference.preprocessImage();
  //ai_inference.normalizeImage();
  //ai_inference.copyImageToInputTensor();
  //ai_inference.runInference();
  
  return 0;
}

