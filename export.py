import torch
import torch.nn as nn
import pnnx
import os
from PIL import Image
import numpy as np
import mobileclip

import ncnn


# 定义图像编码器模块
class ClipImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.image_encoder = model.image_encoder

    def forward(self, x):
        return self.image_encoder(x)


# 定义文本编码器模块
class ClipTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding_layer = model.text_encoder.embedding_layer
        self.positional_embedding = model.text_encoder.positional_embedding
        self.embedding_dropout = model.text_encoder.embedding_dropout
        self.transformer = model.text_encoder.transformer
        self.final_layer_norm = model.text_encoder.final_layer_norm
        self.projection_layer = model.text_encoder.projection_layer

        self.causal_masking = model.text_encoder.causal_masking

    def build_attention_mask(self, context_length: int, batch_size: int):
        """Build causal attention mask [batch_size, context_length, context_length]."""
        # Build mask with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(0)  # add dummy batch dimension
        mask = mask.expand(batch_size, -1, -1)
        return mask

    def forward(self, text_tokens, return_all_tokens=True):
        token_emb = self.embedding_layer(text_tokens)
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).to(
                token_emb.dtype
            )
        token_emb = self.embedding_dropout(token_emb)

        if self.causal_masking:
            attn_mask = self.build_attention_mask(
                context_length=text_tokens.shape[1], batch_size=text_tokens.shape[0]
            )
            attn_mask = attn_mask.to(device=token_emb.device, dtype=token_emb.dtype)

            for layer in self.transformer:
                token_emb = layer(token_emb, attn_mask=attn_mask)
        else:
            for layer in self.transformer:
                token_emb = layer(token_emb)

        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            return token_emb

        token_emb = token_emb[
            torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1)
        ]
        token_emb = token_emb @ self.projection_layer
        return token_emb


class ClipProjection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.projection_layer = model.text_encoder.projection_layer

    def forward(self, x):
        return x @ self.projection_layer


def export(model_name, image_path=None, fp16=False):
    # 验证模型名称有效性
    valid_models = [
        'mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2',
        'mobileclip_b', 'mobileclip_blt'
    ]
    if model_name not in valid_models:
        raise ValueError(f"无效的模型名称。可选模型: {', '.join(valid_models)}")

    # 准备模型路径和输出目录
    model_path = f'./checkpoints/{model_name}.pt'
    export_dir = f'./{model_name}_export'
    os.makedirs(export_dir, exist_ok=True)

    # 加载模型和预处理
    model, _, preprocess = mobileclip.create_model_and_transforms(
        model_name, pretrained=model_path
    )
    tokenizer = mobileclip.get_tokenizer(model_name)
    model.eval()

    # 准备图像输入
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
    else:
        # 创建测试图像（红色方块）
        test_img = np.zeros((256, 256, 3), dtype=np.uint8)
        test_img[:128, :128] = [255, 0, 0]  # 左上角红色方块
        image = Image.fromarray(test_img)

    input_image = preprocess(image).unsqueeze(0)

    # 导出图像编码器
    image_encoder = ClipImageEncoder(model)
    pnnx.export(image_encoder,
                f'{export_dir}/image_encoder.pt',
                input_image, fp16=fp16)

    # 导出文本编码器
    text_encoder = ClipTextEncoder(model)
    input_text = tokenizer(["Test"])  # 单样本输入
    pnnx.export(text_encoder,
                f'{export_dir}/text_encoder.pt',
                input_text, fp16=fp16)

    # 导出投影层
    input_embed = torch.ones((1, 1, 512), dtype=torch.float32)
    projection_layer = ClipProjection(model)
    pnnx.export(projection_layer,
                f'{export_dir}/projection_layer.pt',
                input_embed, fp16=fp16)

    print(f"模型已成功导出到: {export_dir}/")
    print(f"图像编码器输入尺寸: {input_image.shape}")
    print(f"文本编码器输入尺寸: {input_text.shape}")
    print(f"投影层输入尺寸: {input_embed.shape}")

    print(f"正在开始验证ncnn模型...")

    # ncnn验证

    # 图像
    with ncnn.Net() as net:
        # 加载图像编码器模型
        net.load_param(f'{export_dir}/image_encoder.ncnn.param')
        net.load_model(f'{export_dir}/image_encoder.ncnn.bin')

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(input_image.squeeze(0).numpy()).clone())
            _, out0 = ex.extract("out0")
            out_image = torch.from_numpy(np.array(out0)).unsqueeze(0)

    print(f"图像编码器输出尺寸: {out_image.shape}")
    # 计算 MSE
    mse_image = torch.mean((out_image - image_encoder(input_image).detach()) ** 2)
    loss = mse_image.item()
    if loss < 1e-4:
        print(f"图像编码器验证通过，MSE: {loss:.12f}")
    else:
        print(f"图像编码器验证失败，MSE: {loss:.12f}")
        raise RuntimeError("图像编码器验证失败！")

    # 文本
    with ncnn.Net() as net:
        # 加载文本编码器模型
        net.load_param(f'{export_dir}/text_encoder.ncnn.param')
        net.load_model(f'{export_dir}/text_encoder.ncnn.bin')

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(input_text.numpy().astype(np.int32)).clone())
            _, out0 = ex.extract("out0")
            out_text = torch.from_numpy(np.array(out0)).unsqueeze(0)

    print(f"文本编码器输出尺寸: {out_text.shape}")
    # 计算 MSE
    mse_text = torch.mean((out_text - text_encoder(input_text).detach()) ** 2)
    loss = mse_text.item()
    if loss < 1e-4:
        print(f"文本编码器验证通过，MSE: {loss:.12f}")
    else:
        print(f"文本编码器验证失败，MSE: {loss:.12f}")
        raise RuntimeError("文本编码器验证失败！")

    # 投影层
    with ncnn.Net() as net:
        # 加载投影层模型
        net.load_param(f'{export_dir}/projection_layer.ncnn.param')
        net.load_model(f'{export_dir}/projection_layer.ncnn.bin')

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(input_embed.numpy()).clone())
            _, out0 = ex.extract("out0")
            out_projection = torch.from_numpy(np.array(out0)).unsqueeze(0)
    print(f"投影层输出尺寸: {out_projection.shape}")
    # 计算 MSE
    mse_projection = torch.mean((out_projection - projection_layer(input_embed).detach()) ** 2)
    loss = mse_projection.item()
    if loss < 1e-4:
        print(f"投影层验证通过，MSE: {loss:.12f}")
    else:
        print(f"投影层验证失败，MSE: {loss:.12f}")
        raise RuntimeError("投影层验证失败！")

    print("ncnn模型验证完成。")


# 使用示例
if __name__ == '__main__':
    # 导出 mobileclip_s0 模型
    export('mobileclip_blt', image_path="docs/fig_accuracy_latency.png")
    # export('mobileclip_s1', image_path="docs/fig_accuracy_latency.png")
    # export('mobileclip_s2', image_path="docs/fig_accuracy_latency.png")
    # export('mobileclip_b', image_path="docs/fig_accuracy_latency.png")
