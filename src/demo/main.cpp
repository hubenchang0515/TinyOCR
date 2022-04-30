#include <cstdio>
#include <opencv2/opencv.hpp>
#include <tinyocr.h>

void drawRect(cv::Mat& img, const cv::RotatedRect& rect)
{
    cv::Point2f points[4];
    rect.points(points);

    for (int i = 0; i < 4; i++)
    {
        cv::line(img, points[i], points[(i+1)%4], cv::Scalar(0.0f, 255.0f, 0.0f));
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: tinyocr-demo <image>\n");
        return 1;
    }
    cv::Mat img = cv::imread(argv[1]);
    TinyOCR::AI ai;

    auto areas = ai.detail(img);

    for (auto& area : areas)
    {
        drawRect(img, area.rect);
        printf("%s\n", area.text.c_str());
    }

    cv::imshow("preview", img);
    cv::waitKey(-1);
    return 0;
}