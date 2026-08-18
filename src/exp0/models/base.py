"""Base InputEmbedWrapper seam for models accepting multi-hot input vectors."""

import torch
import torch.nn as nn


class InputEmbedWrapper(nn.Module):
    """Wrapper that projects multi-hot input vectors into model hidden_size.

    Inputs:
        input_tuples: (batch_size, n_tuples, d_input) float tensor of multi-hot encoded input tuples.
        target_ids: (batch_size, seq_len) long tensor of target sequence token IDs.

    Processing:
        1. input_tuples are mapped to embeddings via learned linear layer `tuple_proj(input_tuples)`.
        2. target_ids are mapped to embeddings via target embedding lookup `target_embed(target_ids)`.
        3. Input tuple embeddings and target embeddings are concatenated along sequence length dimension.
        4. Concatenated embeddings `inputs_embeds` are passed to backbone forward.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_input: int,
        hidden_size: int,
        vocab_size: int,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_input = d_input
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Learned linear projection for input tuples
        self.tuple_proj = nn.Linear(d_input, hidden_size)

        # Token embedding for target sequence tokens
        self.target_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # Output linear classification head
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_tuples: torch.Tensor,  # (B, n, d_input)
        target_ids: torch.Tensor,    # (B, seq_len)
    ) -> torch.Tensor:               # Returns logits of shape (B, seq_len, vocab_size) for target sequence
        B, n, _ = input_tuples.shape
        _, seq_len = target_ids.shape

        tuple_embeds = self.tuple_proj(input_tuples)        # (B, n, hidden_size)
        target_embeds = self.target_embed(target_ids)       # (B, seq_len, hidden_size)

        inputs_embeds = torch.cat([tuple_embeds, target_embeds], dim=1)  # (B, n + seq_len, hidden_size)

        # Forward through backbone taking inputs_embeds
        hidden_states = self.backbone(inputs_embeds=inputs_embeds)  # (B, n + seq_len, hidden_size)

        # We only predict logits for the target sequence (indices n to n + seq_len)
        target_hidden = hidden_states[:, n:, :]  # (B, seq_len, hidden_size)
        logits = self.head(target_hidden)         # (B, seq_len, vocab_size)

        return logits
