cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(gpu_labs CUDA)


set(CMAKE_CUDA_STANDARD 14)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )


set(CMAKE_CUDA_FLAGS "-Xptxas -O2")

add_executable(lab1_1 lab1/hello.cu)
add_executable(lab1_2 lab1/hello_device.cu)
add_executable(lab1_3 lab1/func_with_params.cu)
add_executable(lab1_4 lab1/device_info.cu)
add_executable(lab2 lab2/lab2.cu)
add_executable(lab3_1 lab3/lab3_1.cu)
add_executable(lab3_2 lab3/lab3_2.cu)
add_executable(lab3_3 lab3/lab3_3.cu)
add_executable(lab4 lab4/lab4.cu)
add_executable(lab5 lab5/lab5.cu)
add_executable(lab5_1 lab5/lab5_1.cu)
add_executable(lab6 lab6/lab6.cu)
add_executable(lab7 lab7/lab7.cu)
target_link_libraries(lab7 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )

add_executable(lab7_1 lab7/lab7_1.cu)
target_link_libraries(lab7_1 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )

add_executable(lab7_2 lab7/lab7_2.cu)
target_link_libraries(lab7_2 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )



set(CMAKE_CUDA_FLAGS "-lGL -lglut -lGLU")
set(CMAKE_CXX_FLAGS "-lGL -lglut -lGLU")

set_target_properties(
        lab1_1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab1_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab1_3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(
        lab1_4
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(
        lab2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(
        lab3_1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab3_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab3_3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab4
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab5
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab5_1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab6
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab7
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab7_1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        lab7_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)