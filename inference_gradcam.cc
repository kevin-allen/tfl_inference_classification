#include <cstdio>
#include "opencv2/opencv.hpp"
#include  <filesystem>
#include "AIInferenceBaseConv.h"
#include "AIInferenceClassification.h"
#include <string>

/*
./inference_gradcam ../python/resnet50_conv_base_model_optim_default.tflite ../python/resnet50_classifier_model_optim_default.tflite  ../data/cat.jpg
See https://github.com/kevin-allen/tfl_inference_classification/blob/main/README.md for more information
*/
  

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
namespace fs = std::filesystem; // Create a namespace alias for convenience

int main(int argc, char* argv[]) {

  int num_arg_needed=3; // required
  int non_opt_arguments; // given by the user
  const char* prog_name=argv[0];
  bool with_n_opt = false;
  bool with_l_opt=false;
  std::string label_file;
  int opt;
  while ((opt=getopt(argc, argv, "n,l:")) != -1)
    {

      switch (opt)
	    { 
        case 'n':
          with_n_opt=true;
          break; 
        default :
        case 'l':
          label_file=optarg;
          with_l_opt=true;
          break;
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

  std::string model_base_conv_file = argv[optind];
  std::string model_classifier_head_file = argv[optind+1];
  std::string image_file = argv[optind+2];

  std::cout << "Model base convolution: " << model_base_conv_file << std::endl;
  std::cout << "Model gradcam/classifier head: " << model_classifier_head_file << std::endl;
  std::cout << "Image: " << image_file << std::endl;
  if (with_l_opt){
    std::cout << "Label: " << label_file << std::endl;
  }
  

  // create an instance for our convolutional network
  AIInferenceBaseConv aibc(model_base_conv_file); // create an instance of AIInference and load the model into memory
  aibc.loadImage(image_file);
  aibc.preprocessImage();
  if (with_n_opt){
    aibc.normalizeImage();
  }
  aibc.copyImageToInputTensor();
  aibc.runInference();
  aibc.copyResultTensorToResultArray();



  // we might want a separate class for the classifier head that does gradcam


  // create an instance for our classification network
  // we can pretend it is a normal classifier model. The input has 2048 channels instead of 3. No preprocessing needed.
  // create an instance of AIInference
  AIInferenceClassification ai_classification(model_classifier_head_file); // create an instance of AIInference and load the model into memory
  if (with_l_opt){
    ai_classification.setLabels(label_file);
  }
  ai_classification.setInputTensorFromBaseConvResultArray(aibc.getResultArray());
  //ai_classification.runInference();
  //ai_classification.copyResultTensorToResultArray();
  //ai_classification.getTopResults();
  //ai_classification.printTopResults();

  return 0;
}

