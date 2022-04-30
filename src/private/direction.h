#ifndef TINYOCR_DIRECTION_H
#define TINYOCR_DIRECTION_H

#include <atomic>
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <model/paddle-ocr-direction-opt.h>
#include <model/paddle-ocr-direction-opt.mem.h>
#include <textarea.h>

namespace TinyOCR
{

class Direction
{
public:
    Direction() noexcept;

    bool isReversed(const cv::Mat& image, float threshold=0.8f) const noexcept;

    ncnn::Mat loadImage(const cv::Mat& image) const noexcept;
    ncnn::Mat classify(const ncnn::Mat& input) const noexcept;

private:
    static std::atomic_bool IS_INITED;
    static ncnn::Net NET_MODEL;
};
    
}; // namespace TinyOCR

#endif // TINYOCR_DIRECTION_H