#include <private/recognizer.h>
#include <model/paddle-ocr-keys.h>

namespace TinyOCR
{

std::atomic_bool Recognizer::IS_INITED{false};
ncnn::Net Recognizer::NET_MODEL;

Recognizer::Recognizer() noexcept
{
    if (!IS_INITED)
    {
        NET_MODEL.load_param(paddle_ocr_recognition_opt_param_bin);
        NET_MODEL.load_model(paddle_ocr_recognition_opt_bin);
        IS_INITED = true;
    }
}

std::string Recognizer::getText(const cv::Mat& image, float threshold) const noexcept
{
    ncnn::Mat input = this->loadImage(image);
    ncnn::Mat output = this->recognize(input);
    return this->text(output, threshold);
}

ncnn::Mat Recognizer::loadImage(const cv::Mat& image) const noexcept
{
    cv::Mat resizedImage;
    // 三通道灰度
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_GRAY2BGR, 3);

    // 短边缩放为32的倍数，另一条边进行等比缩放，然后缩放为32的倍数
    int h = 32;
    int w = image.rows >= image.cols ? 32 : image.cols * h / image.rows;
    cv::resize(resizedImage, resizedImage, cv::Size(w, h), 0, 0);

    // 归一化
    // const float mean[] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    // const float stdVal[]  = {1.0f / 0.229f / 255.0f, 1.0f / 0.224 / 255.0f, 1.0f / 0.225f / 255.0f};
    const float mean[] = {127.5f, 127.5f, 127.5f};
    const float stdVal[] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};

    // backbone
    int right = static_cast<int>(std::ceil(w / 32.0f) * 32.0f ) - w;
    cv::copyMakeBorder(resizedImage, resizedImage, 0, 0, 0, right, cv::BORDER_CONSTANT, cv::Scalar(127.5f, 127.5f, 127.5f));

    ncnn::Mat img = ncnn::Mat::from_pixels(resizedImage.data, ncnn::Mat::PIXEL_RGB, resizedImage.cols, resizedImage.rows);
    img.substract_mean_normalize(mean, stdVal);

    return img;
}

ncnn::Mat Recognizer::recognize(const ncnn::Mat& input) const noexcept
{
    ncnn::Mat output;
    ncnn::Extractor rec = NET_MODEL.create_extractor();
    rec.set_light_mode(true);
    rec.input(paddle_ocr_recognition_opt_param_id::LAYER_x, input);
    rec.extract(paddle_ocr_recognition_opt_param_id::BLOB_save_infer_model_scale_0_tmp_1, output);

    // 结果中每行表示一个字
    // 行中的每一个元素表示字典中该位置的概率
    // 行内index为0表示分隔符，1表示字典的第一个字

    return output;
}

std::string Recognizer::text(const ncnn::Mat& rec, float threshold) const noexcept
{
    std::string result;
    size_t lastIndex = 0; // 上一个字的index
    size_t repeat = 0; // 重复次数
    for (size_t row = 0; row < rec.h; row++)
    {
        size_t maxIndex = 0;
        float maxProbability = 0.0f;
        for (size_t col = 0; col < rec.w; col++)
        {
            if (rec.row(row)[col] > maxProbability)
            {
                maxIndex = col;
                maxProbability = rec.row(row)[col];
            }
        }

        /* CTC */
        #ifdef DEBUG
            if (maxIndex - 1 < paddle_ocr_keys::keys.size())
                printf("(%s:%f)", paddle_ocr_keys::keys[maxIndex-1].c_str(), maxProbability);
            else if (maxIndex == 0)
                printf("-\n");
        #endif

        // 连续的相同字符视为一个字符
        if (maxIndex == lastIndex)
        {
            repeat += 1;
            if (repeat < 5)
            {
                continue;
            }
        }

        repeat = 0;

        // 0 为分隔符
        if (maxIndex == 0)
        {
            lastIndex = 0;
            continue;
        }

        // 概率小于阈值，视为识别错误，忽略
        if (maxProbability < threshold)
        {
            continue;
        }

        if (maxIndex - 1 >= paddle_ocr_keys::keys.size())
        {
            result += " ";
            lastIndex = maxIndex;
            continue;
        }
        
        lastIndex = maxIndex;
        result += paddle_ocr_keys::keys[maxIndex-1];
    }

    return result;
}

}; // namespace TinyOCR