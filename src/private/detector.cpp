#include <private/detector.h>

namespace TinyOCR
{

std::atomic_bool Detector::IS_INITED{false};
ncnn::Net Detector::NET_MODEL;

Detector::Detector() noexcept
{
    if (!IS_INITED)
    {
        NET_MODEL.load_param(paddle_ocr_detection_opt_param_bin);
        NET_MODEL.load_model(paddle_ocr_detection_opt_bin);
        IS_INITED = true;
    }
}

std::vector<TextArea> Detector::findTextArea(const cv::Mat& image, float threshold) const noexcept
{
    ncnn::Mat input = this->loadImage(image);
    ncnn::Mat output = this->detect(input);
    return this->textArea(image, output, threshold);
}

ncnn::Mat Detector::loadImage(const cv::Mat& image) const noexcept
{
    // 长宽缩放为32的倍数
    int w = std::max(int(round(image.cols / 32) * 32), 32);
    int h = std::max(int(round(image.rows / 32) * 32), 32);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(w, h));

    // 归一化
    // 这几个值是 https://image-net.org/ 根据数百万张图片计算得出的 
    // 参考：https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    // PaddleOCR 没有根据自己的训练集计算，直接使用了这些值
    // 参考：https://github.com/PaddlePaddle/PaddleOCR/blob/95c670faf6cf4551c841764cde43a4f4d9d5e634/configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml#L95
    const float mean[] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float stdVal[]  = {1.0f / 0.229f / 255.0f, 1.0f / 0.224 / 255.0f, 1.0f / 0.225f / 255.0f};
    ncnn::Mat img = ncnn::Mat::from_pixels(resizedImage.data, ncnn::Mat::PIXEL_RGB, resizedImage.cols, resizedImage.rows);
    img.substract_mean_normalize(mean, stdVal);

    return img;
}


ncnn::Mat Detector::detect(const ncnn::Mat& input) const noexcept
{
    ncnn::Mat output;
    ncnn::Extractor det = NET_MODEL.create_extractor();
    det.set_light_mode(true);
    det.input(paddle_ocr_detection_opt_param_id::LAYER_x, input);
    det.extract(paddle_ocr_detection_opt_param_id::BLOB_save_infer_model_scale_0_tmp_1, output);

    return output;
}

std::vector<TextArea> Detector::textArea(const cv::Mat& image, const ncnn::Mat& det,float threshold) const noexcept
{
    cv::Mat detection(cv::Size(det.w, det.h), CV_32FC1, det.data);
    cv::dilate(detection, detection, cv::Mat::ones(2, 2, CV_32FC1));

    // findContours 只支持 CV_8UC1 格式
    // cv::Mat binImg = cv::Mat::zeros(detection.size(), CV_8UC1);
    // detection.convertTo(binImg, CV_8UC1);
    cv::Mat binImg = detection >= threshold;
    cv::resize(binImg, binImg, image.size());

    // 寻找轮廓
    // 参考：https://docs.opencv.org/4.x/df/d0d/tutorial_find_contours.html
    // findContours 只支持 CV_8UC1 格式
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binImg, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<TextArea> results;
    for(size_t i = 0; i < contours.size(); i++)
    {
        cv::RotatedRect box = cv::minAreaRect(contours[i]);
        if (box.size.width <= 4 || box.size.height <= 4)
            continue;
        
        TextArea textArea;
        textArea.rect = dilate(box);
        textArea.image = clip(image, textArea.rect);
        
        // 旋转矫正
        if (textArea.image.rows > textArea.image.cols)
        {
            cv::transpose(textArea.image, textArea.image);
            cv::flip(textArea.image, textArea.image, 0);
        }

        results.push_back(textArea);
    }
    return results;
}

cv::RotatedRect Detector::dilate(const cv::RotatedRect& box, float r) const noexcept
{
    // Vatti Clip 的多边形扩张算法经验公式
    // D = S * (1-r^2) / L
    // D : 扩张距离
    // S : 原矩形面积
    // L : 原矩形周长
    // r : 是经验参数，通常取0.4
    float S = box.size.width * box.size.height;
    float L = 2.0f * (box.size.width + box.size.height);
    float D = S * (1.0f - r * r) / L;
    return cv::RotatedRect(box.center, cv::Size2f(box.size.width + 2 * D, box.size.height + 2 * D), box.angle);
}


cv::Mat Detector::clip(const cv::Mat& input, const cv::RotatedRect& box) const noexcept
{
    std::vector<cv::Point2f> points(4);
        box.points(points.data());
        cv::Rect rect = cv::boundingRect(points);

        cv::Point2f center((2.0f*rect.x + rect.width) / 2.0f, (2.0f*rect.y + rect.height) / 2.0f);
        cv::Size size(rect.width, rect.height);
        cv::Mat output;
        cv::getRectSubPix(input, size, center, output);

        int side = std::sqrt(rect.width*rect.width + rect.height*rect.height);
        cv::copyMakeBorder(output, output, (side-rect.height)/2, (side-rect.height)/2, (side-rect.width)/2, (side-rect.width)/2, cv::BORDER_ISOLATED);
        cv::Mat affineMat = cv::getRotationMatrix2D(cv::Point2f(side/2, side/2), box.angle, 1.0);
        cv::warpAffine(output, output, affineMat, cv::Size(side, side), cv::INTER_CUBIC);
        cv::getRectSubPix(output, box.size, cv::Point2f(side/2, side/2), output);

        return output;
}
    
}; // namespace TinyOCR