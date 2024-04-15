#ifndef AIINFERENCECLASSIFICATION_H
#define AIINFERENCECLASSIFICATION_H


#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "opencv2/opencv.hpp"
#include "AIInference.h"




/*
    Typical usage:

    AIInferenceClassification ai_classification(model_file);
    ai_classification.loadImage(image_file);
    ai_classification.preprocessImage();
    ai_classification.normalizeImage();
    ai_classification.copyImageToInputTensor();
    ai_classification.runInference();
    ai_classification.copyResultTensorToResultArray();
    ai_classification.getTopResults();
    ai_classification.printTopResults();        
*/

class AIInferenceClassification: public AIInference{
    /*
    Class to perform inference using TF Lite models with classification head
    Inherits from AIInference
    All the methods related to the classification head are implemented here
    */
    private:
        int nResults = 5;
        std::string label_file_;
        std::vector<std::string> labels;
        std::vector<std::pair<float, int>> top_results;
        float threshold = 0.001;
        float* resultArray = nullptr; // array to store the output of the model
        void allocateMemoryForResultArray();

    public:
        AIInferenceClassification(std::string model_file);
        ~AIInferenceClassification();

        void setLabels(std::string label_file);
        void printResults();
        void copyResultTensorToResultArray();
        void getTopResults();
        void printTopResults();        
};

#endif // AIINFERENCECLASSIFICATION_H