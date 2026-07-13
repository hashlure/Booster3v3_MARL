import json
import sys

import torch


state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print("\n".join(f"{key} {tuple(value.shape)}" for key, value in state.items()))
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump({key: value.tolist() for key, value in state.items()}, output, separators=(",", ":"))
