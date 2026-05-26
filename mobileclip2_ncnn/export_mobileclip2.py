import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


MODEL_SPECS = {
    "mobileclip2_s0": {
        "open_clip_name": "MobileCLIP2-S0",
        "checkpoint": "checkpoints/mobileclip2_s0.pt",
        "image_size": 256,
    },
    "mobileclip2_s2": {
        "open_clip_name": "MobileCLIP2-S2",
        "checkpoint": "checkpoints/mobileclip2_s2.pt",
        "image_size": 256,
    },
}

APPLE_VERIFY_TEXTS = ["a diagram", "a dog", "a cat"]
MODULE_MSE_THRESHOLD = 1e-4
APPLE_PROB_DIFF_THRESHOLD = 1e-4


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ImageEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image, normalize=False)


class TextEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.text = model.text

        if self.text.cls_emb is not None or self.text.use_pad_mask:
            raise ValueError("当前导出器只支持 MobileCLIP2 S0/S2 的无 cls/pad-mask 文本塔。")

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        seq_len = text_tokens.shape[1]
        token_emb = self.text.token_embedding(text_tokens)
        token_emb = token_emb + self.text.positional_embedding[:seq_len]
        token_emb = self.text.transformer(token_emb, attn_mask=self.text.attn_mask)
        return self.text.ln_final(token_emb)


class TextProjection(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.projection = model.text.text_projection

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        return text_embed @ self.projection


def _load_image(path: Path | None, size: int) -> Image.Image:
    if path is not None and path.exists():
        return Image.open(path).convert("RGB")

    test_img = np.zeros((size, size, 3), dtype=np.uint8)
    test_img[: size // 2, : size // 2] = [255, 0, 0]
    return Image.fromarray(test_img)


def _tool_path(path: Path) -> str:
    return path.resolve().as_posix()


def _register_local_configs(open_clip_module, config_dir: Path, model_name: str) -> None:
    list_models = getattr(open_clip_module, "list_models", None)
    if list_models is not None and model_name in list_models():
        return

    add_model_config = getattr(open_clip_module, "add_model_config", None)
    if add_model_config is None:
        raise RuntimeError(
            "当前 open_clip 版本找不到 MobileCLIP2，也不支持 add_model_config；"
            "请按 apple/ml-mobileclip README 安装带 MobileCLIP2 的 OpenCLIP。"
        )
    add_model_config(config_dir)


def _replace_text_layer_norms(module: nn.Module) -> None:
    import open_clip

    open_clip_layer_norm = open_clip.transformer.LayerNorm
    for name, child in list(module.named_children()):
        if isinstance(child, open_clip_layer_norm):
            replacement = nn.LayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            _replace_text_layer_norms(child)


def _build_model(model_key: str, checkpoint: Path):
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model

    spec = MODEL_SPECS[model_key]
    config_dir = Path(__file__).resolve().parent / "model_configs"
    _register_local_configs(open_clip, config_dir, spec["open_clip_name"])

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"未找到权重: {checkpoint}. 可用 `hf download apple/{spec['open_clip_name']}` "
            "下载后传入 --checkpoint。"
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        spec["open_clip_name"],
        pretrained=str(checkpoint),
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
    )
    tokenizer = open_clip.get_tokenizer(spec["open_clip_name"])
    model.eval()
    model = reparameterize_model(model).eval()
    _replace_text_layer_norms(model.text)
    return model, preprocess, tokenizer


def _run_ncnn(net_path: Path, model_path: Path, input_array: np.ndarray) -> torch.Tensor:
    import ncnn

    with ncnn.Net() as net:
        net.load_param(_tool_path(net_path))
        net.load_model(_tool_path(model_path))
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(input_array).clone())
            _, out0 = ex.extract("out0")
            return torch.from_numpy(np.array(out0))


def _ncnn_text_feature_for_tokens(output_dir: Path, tokens: torch.Tensor) -> torch.Tensor:
    token_features = _run_ncnn(
        output_dir / "text_encoder.ncnn.param",
        output_dir / "text_encoder.ncnn.bin",
        tokens.unsqueeze(0).cpu().numpy().astype(np.int32),
    ).reshape(1, 77, 512)

    eot_index = int(tokens.argmax(dim=-1))
    projection_input = token_features[:, eot_index, :]
    return _run_ncnn(
        output_dir / "projection_layer.ncnn.param",
        output_dir / "projection_layer.ncnn.bin",
        projection_input.cpu().numpy(),
    ).reshape(1, 512)


def _verify_apple_way(
    output_dir: Path,
    model: nn.Module,
    tokenizer,
    image_input: torch.Tensor,
) -> None:
    text_input = tokenizer(APPLE_VERIFY_TEXTS)

    with torch.no_grad():
        image_features = model.encode_image(image_input, normalize=False).detach().cpu()
        text_features = model.encode_text(text_input, normalize=False).detach().cpu()
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        torch_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    ncnn_image = _run_ncnn(
        output_dir / "image_encoder.ncnn.param",
        output_dir / "image_encoder.ncnn.bin",
        image_input.squeeze(0).cpu().numpy(),
    ).reshape(1, 512)
    ncnn_image = torch.nn.functional.normalize(ncnn_image, dim=-1)

    ncnn_text_features = []
    for tokens in text_input:
        text_feature = _ncnn_text_feature_for_tokens(output_dir, tokens)
        ncnn_text_features.append(text_feature.squeeze(0))
    ncnn_text_features = torch.stack(ncnn_text_features, dim=0)
    ncnn_text_features = torch.nn.functional.normalize(ncnn_text_features, dim=-1)
    torch_logits = 100.0 * image_features @ text_features.T
    ncnn_logits = 100.0 * ncnn_image @ ncnn_text_features.T
    ncnn_probs = ncnn_logits.softmax(dim=-1)

    prob_diff = (torch_probs - ncnn_probs).abs()
    max_prob_diff = prob_diff.max().item()
    max_logit_diff = (torch_logits - ncnn_logits).abs().max().item()
    image_feature_mse = torch.mean((image_features - ncnn_image) ** 2).item()
    text_feature_mse = torch.mean((text_features - ncnn_text_features) ** 2).item()

    print("Apple-way 端到端概率对比:")
    for index, label in enumerate(APPLE_VERIFY_TEXTS):
        print(
            f"  {label}: torch={torch_probs[0, index].item():.8f}, "
            f"ncnn={ncnn_probs[0, index].item():.8f}, "
            f"abs_diff={prob_diff[0, index].item():.8f}"
        )
    print(f"Apple-way 归一化图像特征 MSE: {image_feature_mse:.12f}")
    print(f"Apple-way 归一化文本特征 MSE: {text_feature_mse:.12f}")
    print(f"Apple-way 最大 logit 差: {max_logit_diff:.12f}")
    print(f"Apple-way 最大概率差: {max_prob_diff:.12f}")

    if max_prob_diff >= APPLE_PROB_DIFF_THRESHOLD:
        raise RuntimeError(
            f"Apple-way 端到端验证失败，最大概率差 {max_prob_diff:.12f} "
            f">= {APPLE_PROB_DIFF_THRESHOLD}。"
        )


def _verify_export(
    output_dir: Path,
    model: nn.Module,
    tokenizer,
    image_encoder: nn.Module,
    text_encoder: nn.Module,
    projection_layer: nn.Module,
    image_input: torch.Tensor,
    text_input: torch.Tensor,
) -> None:
    with torch.no_grad():
        expected_image = image_encoder(image_input).detach().cpu()
        expected_text_tokens = text_encoder(text_input).detach().cpu()
        eot_index = int(text_input.argmax(dim=-1)[0])
        projection_input = expected_text_tokens[:, eot_index, :]
        expected_projection = projection_layer(projection_input).detach().cpu()

    ncnn_image = _run_ncnn(
        output_dir / "image_encoder.ncnn.param",
        output_dir / "image_encoder.ncnn.bin",
        image_input.squeeze(0).cpu().numpy(),
    ).reshape(expected_image.shape)
    image_mse = torch.mean((ncnn_image - expected_image) ** 2).item()

    ncnn_text = _run_ncnn(
        output_dir / "text_encoder.ncnn.param",
        output_dir / "text_encoder.ncnn.bin",
        text_input.cpu().numpy().astype(np.int32),
    ).reshape(expected_text_tokens.shape)
    text_mse = torch.mean((ncnn_text - expected_text_tokens) ** 2).item()

    ncnn_projection = _run_ncnn(
        output_dir / "projection_layer.ncnn.param",
        output_dir / "projection_layer.ncnn.bin",
        projection_input.cpu().numpy(),
    ).reshape(expected_projection.shape)
    projection_mse = torch.mean((ncnn_projection - expected_projection) ** 2).item()

    print(f"图像编码器输出尺寸: {tuple(expected_image.shape)}, MSE: {image_mse:.12f}")
    print(f"文本编码器输出尺寸: {tuple(expected_text_tokens.shape)}, MSE: {text_mse:.12f}")
    print(f"投影层输出尺寸: {tuple(expected_projection.shape)}, MSE: {projection_mse:.12f}")

    if (
        image_mse >= MODULE_MSE_THRESHOLD
        or text_mse >= MODULE_MSE_THRESHOLD
        or projection_mse >= MODULE_MSE_THRESHOLD
    ):
        raise RuntimeError(f"ncnn 子模块验证失败，MSE 超过阈值 {MODULE_MSE_THRESHOLD}。")

    _verify_apple_way(output_dir, model, tokenizer, image_input)


def export(
    model_key: str,
    checkpoint: Path,
    output_dir: Path,
    image_path: Path | None = None,
    fp16: bool = False,
    verify: bool = True,
) -> None:
    import pnnx

    spec = MODEL_SPECS[model_key]
    output_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess, tokenizer = _build_model(model_key, checkpoint)
    image = _load_image(image_path, spec["image_size"])
    image_input = preprocess(image).unsqueeze(0)
    text_input = tokenizer(["a diagram"])

    image_encoder = ImageEncoder(model).eval()
    text_encoder = TextEncoder(model).eval()
    projection_layer = TextProjection(model).eval()
    projection_input = torch.ones((1, 512), dtype=torch.float32)

    with torch.no_grad():
        pnnx.export(
            image_encoder,
            _tool_path(output_dir / "image_encoder.pt"),
            image_input,
            fp16=fp16,
        )
        pnnx.export(
            text_encoder,
            _tool_path(output_dir / "text_encoder.pt"),
            text_input,
            fp16=fp16,
        )
        pnnx.export(
            projection_layer,
            _tool_path(output_dir / "projection_layer.pt"),
            projection_input,
            fp16=fp16,
        )

    print(f"模型已导出到: {output_dir}")
    print(f"OpenCLIP 模型名: {spec['open_clip_name']}")
    print(f"图像输入尺寸: {tuple(image_input.shape)}")
    print(f"文本输入尺寸: {tuple(text_input.shape)}")
    print(f"投影层输入尺寸: {tuple(projection_input.shape)}")

    if verify:
        print("开始验证 ncnn 模型...")
        _verify_export(
            output_dir,
            model,
            tokenizer,
            image_encoder,
            text_encoder,
            projection_layer,
            image_input,
            text_input,
        )
        print("ncnn 模型验证完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileCLIP2 S0/S2 to ncnn.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS.keys()),
        default="mobileclip2_s0",
        help="要导出的 MobileCLIP2 变体。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MobileCLIP2 权重路径；默认使用 checkpoints/<model>.pt。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="导出目录；默认 mobileclip2_ncnn/exports/<model>_export。",
    )
    parser.add_argument("--image", type=Path, default=Path("docs/fig_accuracy_latency.png"))
    parser.add_argument("--fp16", action="store_true", help="导出 fp16 ncnn 模型。")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="跳过 Python ncnn 运行结果和 PyTorch 结果的 MSE 验证。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spec = MODEL_SPECS[args.model]
    checkpoint = args.checkpoint or Path(spec["checkpoint"])
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "exports" / f"{args.model}_export"
    )
    export(
        model_key=args.model,
        checkpoint=checkpoint,
        output_dir=output_dir,
        image_path=args.image,
        fp16=args.fp16,
        verify=not args.skip_verify,
    )
