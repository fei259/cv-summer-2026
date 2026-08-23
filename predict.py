import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from data.cifar10 import CIFAR10_MEAN, CIFAR10_STD
from models.simple_cnn import SimpleCNN


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "formal_validation"
    / "100pct"
    / "none"
    / "dropout_0p3"
    / "weight_decay_0p0"
    / "seed_123"
    / "best_model.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用训练完成的 SimpleCNN 预测单张图片"
    )

    parser.add_argument(
        "image_path",
        type=Path,
        help="需要预测的图片路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="模型训练时使用的 Dropout 概率",
    )

    return parser.parse_args()


def create_inference_transform():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def main():
    args = parse_args()

    if not args.image_path.is_file():
        raise FileNotFoundError(
            f"图片不存在：{args.image_path}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"模型 checkpoint 不存在：{args.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 创建用于推理的图像预处理 transform
    transform = create_inference_transform()

    with Image.open(args.image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image)

    # 增加 batch 维度
    image_tensor = image_tensor.unsqueeze(0).to(device)

    model = SimpleCNN(
        dropout_rate=args.dropout,
    ).to(device)

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        # 得到类别和置信度
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    class_name = CLASS_NAMES[predicted_index.item()]

    print("使用设备：", device)
    print("输入 Tensor 形状：", image_tensor.shape)
    print("预测类别：", class_name)
    print(f"预测置信度：{confidence.item():.2%}")


if __name__ == "__main__":
    main()