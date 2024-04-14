#include <cstdio>
#include "opencv2/opencv.hpp"
#include  <filesystem>
#include "AIInference.h"
#include <string>

/*
./inference_classification /tmp/mobilenet_v1_1.0_224.tflite ~/repo/tensorflow/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp  -l /tmp/labels.txt 
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
  bool with_l_opt=false;
  std::string label_file;

  int opt;
  while ((opt=getopt(argc, argv, "l:")) != -1)
    {
      switch (opt)
	    {
        case 'l':
          label_file=optarg;
          with_l_opt=true;
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
  if (with_l_opt){
    std::cout << "Label: " << label_file << std::endl;
  }


  // create an instance of AIInference
  AIInference ai_inference(model_file); // create an instance of AIInference and load the model into memory
  ai_inference.set_labels(label_file);
  ai_inference.loadImage(image_file);
  ai_inference.preprocessImage();
  ai_inference.normalizeImage();
  ai_inference.copyImageToInputTensor();

  ai_inference.runInference();
  ai_inference.copyResultTensorToResultArray();
  ai_inference.getTopResults();
  ai_inference.printTopResults();
  return 0;
}
