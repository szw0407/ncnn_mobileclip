# MobileClip NCNN

## 使用方法(导出)
### 下载权重
将下载的权重mobileclip_xxx.pt权重文件放在 checkpoints/ 目录下。

### 模型下载

本项目提供导出好的模型：
[Google Drive](https://drive.google.com/file/d/1WFQEwWxUCFhDASbXv7fAlXUHn1BnVGGI/view?usp=drive_link)

### 导出模型
修改export.py中的代码
```
# 使用示例
if __name__ == '__main__':
    # 导出 mobileclip_s0 模型
    export('mobileclip_blt', image_path="docs/fig_accuracy_latency.png")
```
修改mobileclip_blt为您想要导出的模型名称。
可选：
mobileclip_s0,mobileclip_s1,mobileclip_s2,mobileclip_b,mobileclip_blt

当程序输出
```
模型已成功导出到: ./mobileclip_blt_export/
图像编码器输入尺寸: torch.Size([1, 3, 224, 224])
文本编码器输入尺寸: torch.Size([1, 77])
投影层输入尺寸: torch.Size([1, 1, 512])
正在开始验证ncnn模型...
图像编码器输出尺寸: torch.Size([1, 512])
图像编码器验证通过，MSE: 0.000000021671
文本编码器输出尺寸: torch.Size([1, 77, 512])
文本编码器验证通过，MSE: 0.000000000020
投影层输出尺寸: torch.Size([1, 1, 512])
投影层验证通过，MSE: 0.000000000000
ncnn模型验证完成。
```
说明模型导出成功。

## 使用方法(推理)

cpp推理代码见ncnn_mobileclip_infer
把导出的模型参照代码放在合适位置即可

## MobileCLIP2

MobileCLIP2 的 NCNN 适配放在 `mobileclip2_ncnn/`，当前优先支持 `MobileCLIP2-S0` 和 `MobileCLIP2-S2`。用法见 `mobileclip2_ncnn/README.md`。

## 交流群

QQ群：767178345(计算机视觉交流群)
