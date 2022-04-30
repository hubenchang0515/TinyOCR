#ifndef TINYOCR_H
#define TINYOCR_H

#include <string>
#include <vector>
#include <atomic>
#include <memory>

#include "textarea.h"

namespace TinyOCR
{

class Detector;
class Direction;
class Recognizer;

class AI
{
public:
    AI() noexcept;
    ~AI() noexcept;

    std::string brief(const cv::Mat& image, bool angle=false) const noexcept;
    std::vector<TextArea> detail(const cv::Mat& image, bool angle=false) const noexcept;

private:
    Detector* detector;
    Direction* direction;
    Recognizer* recognizer;
};

}; // namespace TinyOCR

#endif // TINYOCR_H