#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "MobileClip2.h"
#include "SimpleTokenizer.h"

namespace {

std::vector<double> softmax(const std::vector<double>& logits) {
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    std::vector<double> probs;
    probs.reserve(logits.size());

    for (double logit : logits) {
        const double prob = std::exp(logit - max_logit);
        probs.push_back(prob);
        sum += prob;
    }
    for (double& prob : probs) {
        prob /= sum;
    }
    return probs;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_export_dir> <image_path> \"<tag1>\" \"<tag2>\" ..." << std::endl;
        return -1;
    }

    try {
        const std::filesystem::path original_cwd = std::filesystem::current_path();
        const std::filesystem::path exe_dir = std::filesystem::absolute(argv[0]).parent_path();
        const std::string model_dir = std::filesystem::absolute(argv[1]).string();
        const std::string image_path = std::filesystem::absolute(argv[2]).string();

        std::vector<std::string> labels;
        for (int i = 3; i < argc; ++i) {
            labels.emplace_back(argv[i]);
        }

        std::filesystem::current_path(exe_dir);
        SimpleTokenizer tokenizer("bpe_simple_vocab_16e6.txt");
        std::filesystem::current_path(original_cwd);
        MobileClip2 model(model_dir);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: could not open image: " << image_path << std::endl;
            return -1;
        }

        const auto tokenized = tokenizer(labels);
        const cv::Mat image_features = model.encode_image(image);

        std::vector<double> logits;
        logits.reserve(tokenized.size());
        for (const auto& tokens : tokenized) {
            cv::Mat text_features = model.encode_text(tokens);
            logits.push_back(image_features.dot(text_features) * 100.0);
        }

        const std::vector<double> probs = softmax(logits);
        for (size_t i = 0; i < labels.size(); ++i) {
            std::cout << "Text: " << labels[i] << ", Prob: " << probs[i] << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
