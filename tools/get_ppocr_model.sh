#! /usr/bin/bash
PROJECT_DIR="$(pwd)/$(dirname $0)/.."
OUTPUT_DIR="$PROJECT_DIR/src/model/"
TEMP_DIR="_temp2"

export PATH="$PATH:$PROJECT_DIR/usr/bin"

rm -rf $TEMP_DIR && mkdir -p $TEMP_DIR && cd $TEMP_DIR

# 下载模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar

# 解包
tar xvf ch_PP-OCRv2_det_infer.tar
tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar
tar xvf ch_PP-OCRv2_rec_infer.tar

# 转 onnx
paddle2onnx --model_dir ch_PP-OCRv2_det_infer \
            --model_file ch_PP-OCRv2_det_infer/inference.pdmodel \
            --params_file ch_PP-OCRv2_det_infer/inference.pdiparams \
            --save_file paddle-ocr-detection.onnx \
            --opset_version 11 \
            --enable_onnx_checker True

paddle2onnx --model_dir ch_ppocr_mobile_v2.0_cls_infer \
            --model_file ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel \
            --params_file ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams \
            --save_file paddle-ocr-direction.onnx \
            --opset_version 11 \
            --enable_onnx_checker True 

paddle2onnx --model_dir ch_PP-OCRv2_rec_infer \
            --model_file ch_PP-OCRv2_rec_infer/inference.pdmodel \
            --params_file ch_PP-OCRv2_rec_infer/inference.pdiparams \
            --save_file paddle-ocr-recognition.onnx \
            --opset_version 11 \
            --enable_onnx_checker True


# simplify
python3 -m onnxsim paddle-ocr-detection.onnx \
        paddle-ocr-detection-simplify.onnx \
        --input-shape "1,3,960,960"

python3 -m onnxsim paddle-ocr-direction.onnx \
        paddle-ocr-direction-simplify.onnx \
        --input-shape "1,3,48,960"

python3 -m onnxsim paddle-ocr-recognition.onnx \
        paddle-ocr-recognition-simplify.onnx \
        --input-shape "1,3,32,960"


# 转 ncnn 模型
onnx2ncnn paddle-ocr-detection-simplify.onnx \
          paddle-ocr-detection.param \
          paddle-ocr-detection.bin

onnx2ncnn paddle-ocr-direction-simplify.onnx \
          paddle-ocr-direction.param \
          paddle-ocr-direction.bin

onnx2ncnn paddle-ocr-recognition-simplify.onnx \
          paddle-ocr-recognition.param \
          paddle-ocr-recognition.bin


# 优化模型
ncnnoptimize paddle-ocr-detection.param \
             paddle-ocr-detection.bin \
             paddle-ocr-detection-opt.param \
             paddle-ocr-detection-opt.bin \
             1

ncnnoptimize paddle-ocr-direction.param \
             paddle-ocr-direction.bin \
             paddle-ocr-direction-opt.param \
             paddle-ocr-direction-opt.bin \
             1

ncnnoptimize paddle-ocr-recognition.param \
             paddle-ocr-recognition.bin \
             paddle-ocr-recognition-opt.param \
             paddle-ocr-recognition-opt.bin \
             1

# 修整个别 ncnn 支持有问题的算子
SRC="Squeeze                  Squeeze_0                1 1 pool2d_2.tmp_0 squeeze_0.tmp_0 -23300=1,1"
DST="Reshape                  Squeeze_0                1 1 pool2d_2.tmp_0 squeeze_0.tmp_0 0=-1 1=512"
sed -i "s#$SRC#$DST#g" paddle-ocr-recognition-opt.param

# 转头文件
ncnn2mem paddle-ocr-detection-opt.param paddle-ocr-detection-opt.bin paddle-ocr-detection-opt.h paddle-ocr-detection-opt.mem.h
ncnn2mem paddle-ocr-direction-opt.param paddle-ocr-direction-opt.bin paddle-ocr-direction-opt.h paddle-ocr-direction-opt.mem.h
ncnn2mem paddle-ocr-recognition-opt.param paddle-ocr-recognition-opt.bin paddle-ocr-recognition-opt.h paddle-ocr-recognition-opt.mem.h

# 复制到 src/model
mkdir -p $OUTPUT_DIR
cp *.h $OUTPUT_DIR