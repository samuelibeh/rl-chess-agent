import torch

from chessrl.encoding import POLICY_SIZE
from chessrl.network import PolicyValueNet, load_checkpoint, save_checkpoint


def test_output_shapes_and_value_range():
    net = PolicyValueNet(channels=16, blocks=2).eval()
    logits, value = net(torch.randn(5, 19, 8, 8))
    assert logits.shape == (5, POLICY_SIZE)
    assert value.shape == (5,)
    assert value.abs().max() <= 1.0


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1)
    net = PolicyValueNet(channels=16, blocks=2).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(net, path, iteration=7)
    loaded = load_checkpoint(path)
    x = torch.randn(3, 19, 8, 8)
    for a, b in zip(net(x), loaded(x)):
        assert torch.allclose(a, b)
    assert loaded.config() == {"channels": 16, "blocks": 2}
