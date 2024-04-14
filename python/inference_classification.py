import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys, getopt



def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hm:i:l:",["model=","image=","labels="])
    except getopt.GetoptError:
        print ('inference_classification.py -m <modelfile> -i <imagefile> -l <labelfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('inference_classification.py -m <modelfile> -i <imagefile> -l <labelfile>')
            sys.exit()
        elif opt in ("-m", "--model"):
            modelfile = arg
        elif opt in ("-i", "--image"):
            imagefile = arg
        elif opt in ("-l", "--labels"):
            labelfile = arg


    print ('Model file:', modelfile)
    print ('Image file:', imagefile)
    print ('Label file:', labelfile)


    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=modelfile)
    interpreter.allocate_tensors()

    # load image and preprocess
    image = cv2.imread(imagefile)
    # print mean value for RGB channels
    print(np.mean(image, axis=(0,1)))
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    # print mean value for RGB channels
    print(np.mean(image, axis=(0,1)))
    image = np.expand_dims(image, axis=0)
    image = (image-127.5)/127.5
    # print mean value for RGB channels
    print(np.mean(image, axis=(0,1,2)))


    # read label file

    labels = []
    with open(labelfile, "r") as f:
        for line in f:
            labels.append(line.strip())

    # perform inference with the tensorflow Lite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # plot the 5 labels with the highest probability
    print("")
    print("Results:")

    top5 = np.argsort(output_data[0])[-5:][::-1]
    for i in top5:
        print(labels[i], output_data[0][i])

    
if __name__ == "__main__":
   main(sys.argv[1:])






