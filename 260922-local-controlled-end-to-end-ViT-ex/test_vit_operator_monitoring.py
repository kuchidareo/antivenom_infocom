import unittest
import warnings

import torch

from models import get_model


class TinyViTOperatorTests(unittest.TestCase):
    def build_model(self) -> torch.nn.Module:
        return get_model(
            "tiny_vit",
            num_classes=10,
            input_size=(32, 32),
            batch_size=2,
            model_depth=4,
            vit_patch_size=4,
            vit_embed_dim=128,
            vit_heads=4,
            vit_mlp_ratio=2,
        )

    def test_operator_inventory(self) -> None:
        model = self.build_model()
        operators = [
            (name, type(module).__name__)
            for name, module in model.named_modules()
            if name and not any(module.children())
        ]
        self.assertEqual(len(operators), 55)
        self.assertIn(("blocks.0.attn.qk_matmul", "QueryKeyMatMul"), operators)
        self.assertIn(("blocks.0.attn.softmax", "Softmax"), operators)
        self.assertIn(("blocks.0.attn.av_matmul", "AttentionValueMatMul"), operators)
        self.assertIn(("blocks.0.attention_residual", "ResidualAdd"), operators)
        self.assertIn(("blocks.0.gelu", "GELU"), operators)

    def test_every_operator_runs_forward_and_backward_once(self) -> None:
        model = self.build_model()
        operators = {
            name: module
            for name, module in model.named_modules()
            if name and not any(module.children())
        }
        forward_counts = dict.fromkeys(operators, 0)
        backward_counts = dict.fromkeys(operators, 0)
        handles = []
        for name, module in operators.items():
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, _output, name=name: forward_counts.__setitem__(
                        name, forward_counts[name] + 1
                    )
                )
            )
            handles.append(
                module.register_full_backward_hook(
                    lambda _module, _grad_input, _grad_output, name=name: backward_counts.__setitem__(
                        name, backward_counts[name] + 1
                    )
                )
            )
        try:
            output = model(torch.randn(2, 3, 32, 32))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Full backward hook is firing when gradients are computed",
                )
                output.sum().backward()
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertEqual(set(forward_counts.values()), {1})
        self.assertEqual(set(backward_counts.values()), {1})


if __name__ == "__main__":
    unittest.main()
