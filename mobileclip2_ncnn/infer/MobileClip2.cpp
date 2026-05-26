#include "MobileClip2.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {

void load_ncnn(ncnn::Net& net, const std::filesystem::path& param_path, const std::filesystem::path& bin_path) {
    if (net.load_param(param_path.string().c_str()) != 0) {
        throw std::runtime_error("failed to load param: " + param_path.string());
    }
    if (net.load_model(bin_path.string().c_str()) != 0) {
        throw std::runtime_error("failed to load bin: " + bin_path.string());
    }
}

cv::Mat mat_to_row(const ncnn::Mat& mat) {
    const int count = mat.total();
    cv::Mat result(1, count, CV_32F);
    std::memcpy(result.data, mat.data, count * sizeof(float));
    cv::normalize(result, result);
    return result;
}

} // namespace

MobileClip2::MobileClip2(const std::string& model_dir, int image_size) : image_size_(image_size) {
    const std::filesystem::path root(model_dir);
    load_ncnn(
        image_encoder_,
        root / "image_encoder.ncnn.param",
        root / "image_encoder.ncnn.bin"
    );
    load_ncnn(
        text_encoder_,
        root / "text_encoder.ncnn.param",
        root / "text_encoder.ncnn.bin"
    );
    load_ncnn(
        projection_layer_,
        root / "projection_layer.ncnn.param",
        root / "projection_layer.ncnn.bin"
    );
}

cv::Mat MobileClip2::encode_image(cv::Mat image) {
    if (!image.isContinuous()) {
        image = image.clone();
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        image.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        image.cols,
        image.rows,
        image_size_,
        image_size_
    );

    float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = image_encoder_.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    int eot_token_index = 0;
    for (int i = 0; i < static_cast<int>(tokens.size()) && i < max_length; ++i) {
        if (tokens[i] == 49407) {
            eot_token_index = i;
            break;
        }
    }

    ncnn::Mat text_embed = out.row_range(eot_token_index, 1);
    ncnn::Extractor projection_ex = projection_layer_.create_extractor();
    projection_ex.input("in0", text_embed);

    ncnn::Mat projected;
    projection_ex.extract("out0", projected);
    return mat_to_row(projected);
}

cv::Mat MobileClip2::encode_text(const std::vector<int>& tokens) {
    constexpr int max_length = 77;
    ncnn::Mat in(max_length);
    in.fill(0);
    int* input_ptr = in.row<int>(0);
    for (int i = 0; i < static_cast<int>(tokens.size()) && i < max_length; ++i) {
        input_ptr[i] = tokens[i];
    }

    ncnn::Extractor ex = text_encoder_.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);
    return mat_to_row(out);
}
