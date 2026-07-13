import json
import math
import random
import sys

import torch


def norm(x, w, b):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    scale = 1.0 / math.sqrt(var + 1e-5)
    return [(v - mean) * scale * a + c for v, a, c in zip(x, w, b)]


def linear(x, w, b):
    return [sum(v * a for v, a in zip(x, row)) + c for row, c in zip(w, b)]


def forward(x, w):
    x = norm(x, w["base.feature_norm.weight"], w["base.feature_norm.bias"])
    x = [max(0.0, v) for v in linear(x, w["base.mlp.fc1.0.weight"], w["base.mlp.fc1.0.bias"])]
    x = norm(x, w["base.mlp.fc1.2.weight"], w["base.mlp.fc1.2.bias"])
    x = [max(0.0, v) for v in linear(x, w["base.mlp.fc2.0.0.weight"], w["base.mlp.fc2.0.0.bias"])]
    x = norm(x, w["base.mlp.fc2.0.2.weight"], w["base.mlp.fc2.0.2.bias"])
    return linear(x, w["act.action_out.linear.weight"], w["act.action_out.linear.bias"])


weights = json.load(open(sys.argv[1], encoding="utf-8"))
model = torch.jit.load(sys.argv[2], map_location="cpu")
maximum = 0.0
for seed in range(20):
    random.seed(seed)
    observation = [random.uniform(-1.0, 1.0) for _ in range(59)]
    mask = [1.0 if random.random() > 0.2 else 0.0 for _ in range(22)]
    mask[0] = 1.0
    action, logits = model(torch.tensor([observation]), torch.tensor([mask]))
    light = forward(observation, weights)
    maximum = max(maximum, max(abs(a - b) for a, b in zip(light, logits[0].tolist()) if b > -1e8))
    light_action = max(range(22), key=lambda i: light[i] if mask[i] else -1e9)
    assert light_action == int(action[0]), (seed, light_action, int(action[0]))
print("parity_ok max_logit_error=", maximum)
