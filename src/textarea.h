#ifndef TINYOCR_TEXTAREA_H
#define TINYOCR_TEXTAREA_H

#include <string>
#include <opencv2/opencv.hpp>

namespace TinyOCR
{

struct TextArea
{
    std::string text;
    cv::RotatedRect rect;
    cv::Mat image;
};

}; // namespace TinyOCR

#endif // TINYOCR_TEXTAREA_H