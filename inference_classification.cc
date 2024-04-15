#include <cstdio>
#include "opencv2/opencv.hpp"
#include  <filesystem>
#include "AIInferenceClassification.h"
#include <string>

/*
./inference_classification /tmp/mobilenet_v1_1.0_224.tflite ~/repo/tensorflow/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp  -l /tmp/labels.txt -n 
or
 ./inference_classification ../python/resnet50_model.tflite ../data/cat.jpg  -l ../python/imagenet_labels.txt 

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
  bool with_n_opt=false;
  std::string label_file;

  int opt;
  while ((opt=getopt(argc, argv, "nl:")) != -1)
    {
      switch (opt)
	    {
        case 'n':
          with_n_opt=true;
          break;
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
  AIInferenceClassification ai_classification(model_file); // create an instance of AIInference and load the model into memory
  if (with_l_opt){
    ai_classification.setLabels(label_file);
  }
  ai_classification.loadImage(image_file);
  ai_classification.preprocessImage();
  if (with_n_opt){
  ai_classification.normalizeImage();
  }
  ai_classification.copyImageToInputTensor();

  ai_classification.runInference();
  ai_classification.copyResultTensorToResultArray();
  ai_classification.getTopResults();
  ai_classification.printTopResults();
  return 0;
}

