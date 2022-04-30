# TinyOCR

通过 [ncnn](https://github.com/Tencent/ncnn) 部署的 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 模型

![示例](https://raw.githubusercontent.com/hubenchang0515/resource/master/TinyOCR/demo1.png)

## Build

```
mkdir build 
cd build 
cmake .. -DOCR_USE_GPU=[ON | OFF]
make
```

编译方式需要和 ncnn 一致。即：
* 如果编译 ncnn 时选择了使用 GPU，则 TinyOCR 也应选择使用 GPU
* 如果编译 ncnn 时选择了不使用 GPU，则 TinyOCR 也应选择不使用 GPU

可以运行 `tools` 目录中附带的脚本自动下载 ncnn 并编译

```
mkdir build 
cd build
../tools/get_ncnn.sh
```

* `get_ncnn.sh` 会跟据是否安装了 GPU 的依赖库自动选择是否使用 GPU
* `get_ncnn_cpu.sh` 则始终不使用 GPU
* `get_ppocr_model.sh` 会自动下载 PaddleOCR 的模型并转换为头文件
* `get_ppocr_keys.sh` 会自动下载 PaddleOCR 的字典并转换为头文件

> 由于支持 AVX512 的平台较少，因此上述脚本均禁用了 AVX512