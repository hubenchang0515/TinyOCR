PROJECT_DIR="$(pwd)/$(dirname $0)/.."
OUTPUT_DIR="$PROJECT_DIR/src/model/"
TEMP_DIR="_temp3"

export PATH="$PATH:$PROJECT_DIR/usr/bin"

rm -rf $TEMP_DIR && mkdir -p $TEMP_DIR && cd $TEMP_DIR

# 下载字典
wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.4/ppocr/utils/ppocr_keys_v1.txt

# 转头文件
echo                                                    > paddle-ocr-keys.h.1
echo "#ifndef paddle_ocr_keys_h"                        >> paddle-ocr-keys.h.1
echo "#define paddle_ocr_keys_h"                        >> paddle-ocr-keys.h.1
echo "#include <vector>"                                >> paddle-ocr-keys.h.1
echo "#include <string>"                                >> paddle-ocr-keys.h.1
echo "namespace paddle_ocr_keys{"                       >> paddle-ocr-keys.h.1
echo "std::vector<std::string> keys = {"                >> paddle-ocr-keys.h.1
cat ppocr_keys_v1.txt | awk '{print "\""$0"\""","}'     >> paddle-ocr-keys.h.1
echo "};"                                               >> paddle-ocr-keys.h.1
echo "} // namespace paddle_ocr_keys"                   >> paddle-ocr-keys.h.1
echo "#endif // paddle_ocr_keys_h"                      >> paddle-ocr-keys.h.1
sed 's/"\\"/"\\\\"/g'  paddle-ocr-keys.h.1              > paddle-ocr-keys.h.2
sed 's/\"\"\"/\"\\\"\"/g'  paddle-ocr-keys.h.2          > paddle-ocr-keys.h

# 复制到 src/model
mkdir -p $OUTPUT_DIR
cp *.h $OUTPUT_DIR