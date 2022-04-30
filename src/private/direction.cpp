#include <private/direction.h>

namespace TinyOCR
{

std::atomic_bool Direction::IS_INITED{false};
ncnn::Net Direction::NET_MODEL;

Direction::Direction() noexcept
{
    if (!IS_INITED)
    {
        NET_MODEL.load_param(paddle_ocr_direction_opt_param_bin);
        NET_MODEL.load_model(paddle_ocr_direction_opt_bin);
        IS_INITED = true;
    }
}

bool Direction::isReversed(const cv::Mat& image, float threshold) const noexcept
{
    ncnn::Mat input = this->loadImage(image);
    ncnn::Mat output = this->classify(input);

    return output.row(0)[0] < output.row(0)[1] && output.row(0)[1] >= threshold;
}

ncnn::Mat Direction::loadImage(const cv::Mat& image) const noexcept
{
    cv::Mat resizedImage;
    // 三通道灰度
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_GRAY2BGR, 3);

    // 短边缩放为48的倍数，另一条边进行等比缩放
    int h = 48;
    int w = image.rows >= image.cols ? 48 : image.cols * h / image.rows;
    printf("%d %d\n", h, w);
    cv::resize(resizedImage, resizedImage, cv::Size(w, h), 0, 0);
    
    if (w < 192)
    {
        int right = 192 - w;
        cv::copyMakeBorder(resizedImage, resizedImage, 
                            0, 0, 0, right, 
                            cv::BORDER_CONSTANT,
                            cv::Scalar(127.5f, 127.5f, 127.5f));
    }

    const float mean[] = {127.5f, 127.5f, 127.5f};
    const float stdVal[] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};

    ncnn::Mat img = ncnn::Mat::from_pixels(resizedImage.data, ncnn::Mat::PIXEL_RGB, resizedImage.cols, resizedImage.rows);
    img.substract_mean_normalize(mean, stdVal);

    return img;
}

ncnn::Mat Direction::classify(const ncnn::Mat& input) const noexcept
{
    ncnn::Mat output;
    ncnn::Extractor rec = NET_MODEL.create_extractor();
    rec.set_light_mode(true);
    rec.input(paddle_ocr_direction_opt_param_id::LAYER_x, input);
    rec.extract(paddle_ocr_direction_opt_param_id::BLOB_save_infer_model_scale_0_tmp_1, output);

    return output;
}
    
} // namespace TinyOCR
