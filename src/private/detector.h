#ifndef TINYOCR_DETECTOR_H
#define TINYOCR_DETECTOR_H

#include <atomic>
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <model/paddle-ocr-detection-opt.h>
#include <model/paddle-ocr-detection-opt.mem.h>
#include <textarea.h>

namespace TinyOCR
{

class Detector
{
public:
    Detector() noexcept;

    std::vector<TextArea> findTextArea(const cv::Mat& image, float threshold=0.5f) const noexcept;

    ncnn::Mat loadImage(const cv::Mat& image) const noexcept;
    ncnn::Mat detect(const ncnn::Mat& input) const noexcept;
    std::vector<TextArea> textArea(const cv::Mat& image, const ncnn::Mat& det, float threshold=0.5f) const noexcept;
private:
    static std::atomic_bool IS_INITED;
    static ncnn::Net NET_MODEL;

    cv::RotatedRect dilate(const cv::RotatedRect& box, float r=0.4f) const noexcept;
    cv::Mat clip(const cv::Mat& input, const cv::RotatedRect& box) const noexcept;
};
    
}; // namespace TinyOCR


#endif // TINYOCR_DETECTOR_H