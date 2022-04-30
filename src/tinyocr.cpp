#include <tinyocr.h>
#include <private/detector.h>
#include <private/direction.h>
#include <private/recognizer.h>

namespace TinyOCR
{

AI::AI() noexcept
{
    detector = new Detector;
    direction = new Direction;
    recognizer = new Recognizer;
}

AI::~AI() noexcept
{
    delete detector;
    delete direction;
    delete recognizer;
}

std::string AI::brief(const cv::Mat& image, bool angle) const noexcept
{
    std::string text;
    auto areas = detail(image, angle);
    for (auto& area : areas)
    {
        text += area.text + "\n";
    }

    return text;
}

std::vector<TextArea> AI::detail(const cv::Mat& image, bool angle) const noexcept
{
    auto areas = detector->findTextArea(image, 0.3f);
    for (auto& area : areas)
    {
        if (angle)
        {
            cv::Point2f points[4];
            area.rect.points(points);
            if (direction->isReversed(area.image))
            {
                cv::Mat affineMat = cv::getRotationMatrix2D(cv::Point2f(area.image.cols/2, area.image.rows/2), 180.0f, 1.0f);
                cv::warpAffine(area.image, area.image, affineMat, area.image.size(), cv::INTER_CUBIC);
            }
        }
        
        area.text = recognizer->getText(area.image);
    }

    std::sort(areas.begin(), areas.end(), [](const TinyOCR::TextArea& t1, const TinyOCR::TextArea& t2) -> bool {
        return t1.rect.center.y < t2.rect.center.y  ||  (t1.rect.center.y == t2.rect.center.y && t1.rect.center.x < t2.rect.center.x);
    });
    
    return areas;
}

}; // namespace TinyOCR