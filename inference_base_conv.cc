#include <cstdio>
#include "opencv2/opencv.hpp"
#include  <filesystem>
#include "AIInferenceBaseConv.h"
#include <string>

/*
./inference_base_conv ../python/resnet50_conv_base_model_optim_default.tflite ../data/cat.jpg
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
  bool with_n_opt = false;
  int opt;
  while ((opt=getopt(argc, argv, "n")) != -1)
    {
      switch (opt)
	    { 
        case 'n':
          with_n_opt=true;
          break; 
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
  AIInferenceBaseConv aibc(model_file); // create an instance of AIInference and load the model into memory
  aibc.loadImage(image_file);
  aibc.preprocessImage();
  if (with_n_opt){
    aibc.normalizeImage();
  }
  aibc.copyImageToInputTensor();
  aibc.runInference();
  aibc.copyResultTensorToResultArray();
  
  return 0;
}

