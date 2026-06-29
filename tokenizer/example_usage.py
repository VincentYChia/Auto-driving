from pathlib import Path
from typing import Tuple

import torch
from tokenizer.tokenizer import PointTokenizer


def load_point_tokenizer(weights_path: Path) -> PointTokenizer:
    state_dict = torch.load(str(weights_path), map_location="cpu")

    if "embedding.C" not in state_dict:
        raise KeyError()

    centers = state_dict["embedding.C"]

    # defaults, later gets changed in load state dict
    tokenizer = PointTokenizer(kmeans_or_centers=centers)
    tokenizer.load_state_dict(state_dict, strict=True)
    tokenizer.eval()
    return tokenizer


def demo_round_trip(tokenizer: PointTokenizer, point_xy: Tuple[float, float]) -> None:
    print(f"Input point: {point_xy}")

    embedding, index = tokenizer.encode_points(point_xy)
    print(f"Nearest token index: {int(index)}")
    print(f"Embedding vector shape: {tuple(embedding.shape)}")

    recovered_point = tokenizer.indices_to_points(index)
    print(f"Recovered point: {recovered_point.tolist()}")


if __name__ == "__main__":
    weights_file = Path(__file__).with_name("tokenizer_model.pth")
    if not weights_file.exists():
        raise FileNotFoundError()

    point_tokenizer = load_point_tokenizer(weights_file)
    sample_point = (10.0, -1.2)  # NOTE: any point here!!!
    demo_round_trip(point_tokenizer, sample_point)
