# MobileClip NCNN 导出

## 使用方法
### 下载权重
将下载的权重mobileclip_xxx.pt权重文件放在 checkpoints/ 目录下。
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