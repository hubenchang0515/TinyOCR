#ifndef TINYOCR_RECOGNIZER_H
#define TINYOCR_RECOGNIZER_H

#include <atomic>
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <model/paddle-ocr-recognition-opt.h>
#include <model/paddle-ocr-recognition-opt.mem.h>
#include <textarea.h>

namespace TinyOCR
{

class Recognizer
{
public:
    Recognizer() noexcept;

    std::string getText(const cv::Mat& image, float threshold=0.0f) const noexcept;

    ncnn::Mat loadImage(const cv::Mat& image) const noexcept;
    ncnn::Mat recognize(const ncnn::Mat& input) const noexcept;
    std::string text(const ncnn::Mat& rec, float threshold=0.0f) const noexcept;

private:
    static std::atomic_bool IS_INITED;
    static ncnn::Net NET_MODEL;
};
    
}; // namespace TinyOCR

#endif // TINYOCR_RECOGNIZER_H