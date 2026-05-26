#ifndef MOBILECLIP2_H
#define MOBILECLIP2_H

#include <string>
#include <vector>

#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>

class MobileClip2 {
public:
    explicit MobileClip2(const std::string& model_dir, int image_size = 256);

    cv::Mat encode_image(cv::Mat image);
    cv::Mat encode_text(const std::vector<int>& tokens);

private:
    ncnn::Net image_encoder_;
    ncnn::Net text_encoder_;
    ncnn::Net projection_layer_;
    int image_size_;
};

#endif
