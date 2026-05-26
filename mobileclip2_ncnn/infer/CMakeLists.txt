cmake_minimum_required(VERSION 3.20)
project(ncnn_mobileclip2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NCNN_MOBILECLIP_INFER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../ncnn_mobileclip_infer")

set(ncnn_DIR "${NCNN_MOBILECLIP_INFER_DIR}/3rdparty/ncnn/lib/cmake/ncnn" CACHE PATH "ncnn cmake directory")
set(ncnn_INCLUDE_DIRS "${NCNN_MOBILECLIP_INFER_DIR}/3rdparty/ncnn/include" CACHE PATH "ncnn include directory")
find_package(ncnn REQUIRED)
include_directories(${ncnn_INCLUDE_DIRS})

set(OpenCV_DIR "${NCNN_MOBILECLIP_INFER_DIR}/3rdparty/ocvm/x64/vc17/staticlib" CACHE PATH "OpenCV cmake directory")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ncnn_mobileclip2
    main.cpp
    MobileClip2.cpp
    "${NCNN_MOBILECLIP_INFER_DIR}/SimpleTokenizer.cpp"
)

target_include_directories(ncnn_mobileclip2 PRIVATE
    "${NCNN_MOBILECLIP_INFER_DIR}"
)

target_link_libraries(ncnn_mobileclip2
    ncnn
    ${OpenCV_LIBS}
)

add_custom_command(TARGET ncnn_mobileclip2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NCNN_MOBILECLIP_INFER_DIR}/bpe_simple_vocab_16e6.txt"
        "$<TARGET_FILE_DIR:ncnn_mobileclip2>/bpe_simple_vocab_16e6.txt"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:ncnn_mobileclip2>/../models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NCNN_MOBILECLIP_INFER_DIR}/vocab.txt"
        "$<TARGET_FILE_DIR:ncnn_mobileclip2>/../models/vocab.txt"
)
