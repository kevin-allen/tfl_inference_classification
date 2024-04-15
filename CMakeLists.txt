#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.16)
project(inference_classification C CXX)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


## could we not just add the location of what we need instead of rebuilding these all the time

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "/home/kevin/repo/tensorflow" #"${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)




set(CMAKE_CXX_STANDARD 17)

add_executable(inference_classification
  inference_classification.cc
  AIInference.cc
  AIInferenceClassification.cc
)

add_executable(inference_base_conv
  inference_base_conv.cc
  AIInference.cc
  AIInferenceBaseConv.cc
)




target_link_libraries(inference_classification
  tensorflow-lite
  ${OpenCV_LIBS} 
  )
target_link_libraries(inference_base_conv
  tensorflow-lite
  ${OpenCV_LIBS} 
  )
