#ifndef AIINFERENCEBASECONV_H
#define AIINFERENCEBASECONV_H


#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "opencv2/opencv.hpp"
#include "AIInference.h"




/*
    Typical usage:

    AIInferenceBaseConv ai_bc(model_file);
    ai_bc.loadImage(image_file);
    ai_bc.preprocessImage();
    ai_bc.normalizeImage();
    ai_bc.copyImageToInputTensor();
    ai_bc.runInference();
*/

class AIInferenceBaseConv: public AIInference{
    /*
    Class to perform inference using TF Lite models with classification head
    Inherits from AIInference
    All the methods related to the classification head are implemented here
    */
    private:
        float* resultArray = nullptr; // array to store the output of the model
        void allocate_memory_for_result_array();

    public:
        AIInferenceBaseConv(std::string model_file);
        ~AIInferenceBaseConv();

};

#endif // AIINFERENCECLASSIFICATION_H