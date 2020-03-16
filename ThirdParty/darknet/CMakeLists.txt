##---------------------------------------------------------------------------------------------------------------------
##  RGBD_TOOLS
##---------------------------------------------------------------------------------------------------------------------
##  Copyright 2018 Pablo Ramon Soria (a.k.a. Bardo91) pabramsor@gmail.com
##---------------------------------------------------------------------------------------------------------------------
##  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
##  and associated documentation files (the "Software"), to deal in the Software without restriction,
##  including without limitation the rights to use, copy, modify, merge, publish, distribute,
##  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
##  furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in all copies or substantial
##  portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
##  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
##  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
##  OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
##  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
##---------------------------------------------------------------------------------------------------------------------

cmake_minimum_required (VERSION 3.12 FATAL_ERROR)
project(darknet VERSION 0.3)

##########################################
######		Configure project		######
##########################################

#########################################
###### Loading 3rd party libraries.######
#########################################

FIND_PACKAGE(OpenCV REQUIRED)

FIND_PACKAGE(CUDA REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")

set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES}) # WTF hack
set(CUDA_CUBLAS_LIBRARIES PUBLIC ${CUDA_CUBLAS_LIBRARIES}) # WTF hack
set(CUDA_curand_LIBRARY PUBLIC ${CUDA_curand_LIBRARY}) # WTF hack
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    #-gencode arch=compute_20,code=[sm_20,sm_21] 666 CHECK THIS DEPEND ON THE PLATFORM
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )


#########################################
######			Library 		   ######
#########################################
#add_definitions(-fopenmp)

# Collect all sources
file(GLOB_RECURSE DARKNET_SOURCE_FILES "src/*.cpp" "src/*.c")
file(GLOB_RECURSE DARKNET_HEADER_FILES "include/*.h" "include/*.inl" "src/*.h")
file(GLOB_RECURSE DARKNET_CUDA_SOURCE_FILES "src/*.cu")
 
cuda_add_library(${PROJECT_NAME} SHARED ${DARKNET_HEADER_FILES} ${DARKNET_SOURCE_FILES} ${DARKNET_CUDA_SOURCE_FILES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:src>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC CUDNN GPU OPENCV)
#target_compile_options(${PROJECT_NAME} PUBLIC -lm -pthread)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/include/)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_curand_LIBRARY})

target_link_libraries(${PROJECT_NAME} INTERFACE cudnn)

# install darknet files too
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX} FILES_MATCHING PATTERN "*.h")
install(DIRECTORY src/darknet DESTINATION ${CMAKE_INSTALL_PREFIX}/include FILES_MATCHING PATTERN "*.h")

##########################################
#### BUILD APP
##########################################
file(GLOB_RECURSE DARKNET_EXAMPLE_SOURCE_FILES "examples/*.c")

cuda_add_executable(${PROJECT_NAME}_application ${DARKNET_EXAMPLE_SOURCE_FILES})
target_include_directories(${PROJECT_NAME}_application PRIVATE "include") 

target_link_libraries(${PROJECT_NAME}_application LINK_PRIVATE darknet)

