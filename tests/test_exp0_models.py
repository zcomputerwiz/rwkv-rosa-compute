"""Unit tests for Milestone 2: Model adapters and backbones (Llama and RWKV-7)."""

import torch

from exp0.models.base import InputEmbedWrapper
from exp0.models.llama import LlamaBackbone
from exp0.models.rwkv import RWKV7Backbone


def test_llama_model_forward():
    d_input = 36  # length=6, dim=3 -> 10*3 + 6 = 36
    hidden_size = 384
    vocab_size = 50
    batch_size = 2
    n_tuples = 6
    seq_len = 10

    backbone = LlamaBackbone(hidden_size=hidden_size, num_layers=2, num_heads=4, intermediate_size=512)
    model = InputEmbedWrapper(backbone=backbone, d_input=d_input, hidden_size=hidden_size, vocab_size=vocab_size)

    input_tuples = torch.randn(batch_size, n_tuples, d_input)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_tuples, target_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_rwkv_model_forward():
    d_input = 36
    hidden_size = 384
    vocab_size = 50
    batch_size = 2
    n_tuples = 6
    seq_len = 10

    backbone = RWKV7Backbone(hidden_size=hidden_size, num_layers=2, intermediate_size=512, head_dim=64)
    model = InputEmbedWrapper(backbone=backbone, d_input=d_input, hidden_size=hidden_size, vocab_size=vocab_size)

    input_tuples = torch.randn(batch_size, n_tuples, d_input)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_tuples, target_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_parameter_counts():
    d_input = 36
    hidden_size = 384
    vocab_size = 50

    llama_backbone = LlamaBackbone(hidden_size=hidden_size, num_layers=4, num_heads=6, intermediate_size=1536)
    llama_model = InputEmbedWrapper(backbone=llama_backbone, d_input=d_input, hidden_size=hidden_size, vocab_size=vocab_size)

    rwkv_backbone = RWKV7Backbone(hidden_size=hidden_size, num_layers=4, intermediate_size=1536, head_dim=64)
    rwkv_model = InputEmbedWrapper(backbone=rwkv_backbone, d_input=d_input, hidden_size=hidden_size, vocab_size=vocab_size)

    llama_params = sum(p.numel() for p in llama_model.parameters())
    rwkv_params = sum(p.numel() for p in rwkv_model.parameters())

    assert llama_params > 0
    assert rwkv_params > 0
    print(f"Llama model total params: {llama_params:,}")
    print(f"RWKV-7 model total params: {rwkv_params:,}")
