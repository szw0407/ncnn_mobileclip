cmake_minimum_required(VERSION 3.31)
project(ncnn_mobileclip_full)

set(CMAKE_CXX_STANDARD 20)

# 导入ncnn
set(ncnn_DIR 3rdparty/ncnn/lib/cmake/ncnn)
set(ncnn_INCLUDE_DIRS 3rdparty/ncnn/include)

find_package(ncnn REQUIRED)    # 找到ncnn库
include_directories(${ncnn_INCLUDE_DIRS})

# 导入 OpenCV
set(OpenCV_DIR 3rdparty/ocvm/x64/vc17/staticlib)
find_package(OpenCV REQUIRED) # 找到opencv库
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(ncnn_mobileclip_full main.cpp
        MobileClip.cpp
        SimpleTokenizer.cpp
)

target_link_libraries(ncnn_mobileclip_full
    ncnn
    ${OpenCV_LIBS}
)