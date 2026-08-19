"""Base input seam for models accepting Match-3 multi-hot features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedWrapper(nn.Module):
    """Project tuple and continuation features through one shared input layer.

    Pfau et al.'s Match-3 implementation feeds both the initial multi-hot tuple
    vectors and later continuation-token feature vectors through one
    ``nn.Linear`` input adapter. In particular, reduced CoT tuple-index and digit
    tokens reuse columns that also encode the original tuple positions/digits.

    ``target_feature_indices`` maps each target vocabulary id onto the shared
    input feature column that should represent that token. Tokens without a
    tuple-feature analogue occupy dedicated columns after ``d_input``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_input: int,
        hidden_size: int,
        vocab_size: int,
        *,
        target_feature_indices: torch.Tensor | None = None,
        input_feature_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_input = d_input
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        if target_feature_indices is None:
            target_feature_indices = torch.arange(
                d_input,
                d_input + vocab_size,
                dtype=torch.long,
            )
            if input_feature_dim is None:
                input_feature_dim = d_input + vocab_size
        else:
            if target_feature_indices.dtype != torch.long:
                target_feature_indices = target_feature_indices.to(torch.long)
            if target_feature_indices.ndim != 1:
                raise ValueError("target_feature_indices must be rank-1.")
            if target_feature_indices.numel() != vocab_size:
                raise ValueError(
                    "target_feature_indices must contain one entry per vocab id."
                )
            if input_feature_dim is None:
                input_feature_dim = int(target_feature_indices.max().item()) + 1

        assert input_feature_dim is not None
        if input_feature_dim < d_input:
            raise ValueError("input_feature_dim must cover all tuple features.")
        if target_feature_indices.numel() and (
            int(target_feature_indices.min().item()) < 0
            or int(target_feature_indices.max().item()) >= input_feature_dim
        ):
            raise ValueError("target feature mapping is outside input_feature_dim.")

        self.input_feature_dim = input_feature_dim
        self.register_buffer(
            "target_feature_indices",
            target_feature_indices,
            persistent=True,
        )

        # Keep PyTorch's default nn.Linear initialization here deliberately. The
        # authors' InputEmbedCausalTransformer likewise adds a default-initialized
        # input linear around the normally initialized Llama backbone.
        self.input_proj = nn.Linear(input_feature_dim, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _tuple_hidden(self, input_tuples: torch.Tensor) -> torch.Tensor:
        # Tuple vectors occupy the first d_input feature columns. Applying the
        # corresponding weight slice is algebraically identical to zero-padding
        # the tuple vectors to input_feature_dim and calling input_proj.
        weight = self.input_proj.weight[:, : self.d_input]
        return F.linear(input_tuples, weight, self.input_proj.bias)

    def _target_hidden(self, target_ids: torch.Tensor) -> torch.Tensor:
        mapped = self.target_feature_indices[target_ids]
        # One-hot feature vector @ input_proj.weight.T is exactly an embedding
        # lookup into the transposed shared linear weight. The same bias is then
        # added at every continuation position, matching nn.Linear semantics.
        hidden = F.embedding(mapped, self.input_proj.weight.transpose(0, 1))
        return hidden + self.input_proj.bias

    def target_hidden_states(
        self,
        input_tuples: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return backbone hidden states aligned with target-token positions."""
        _, n, _ = input_tuples.shape

        tuple_embeds = self._tuple_hidden(input_tuples)
        target_embeds = self._target_hidden(target_ids)
        inputs_embeds = torch.cat([tuple_embeds, target_embeds], dim=1)
        hidden_states = self.backbone(inputs_embeds=inputs_embeds)
        return hidden_states[:, n:, :]

    def loss_logits(
        self,
        input_tuples: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Project only positions that participate in next-token loss."""
        target_hidden = self.target_hidden_states(input_tuples, target_ids)
        return self.head(target_hidden[:, :-1, :])

    def answer_logits(
        self,
        input_tuples: torch.Tensor,
        target_ids: torch.Tensor,
        answer_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Project only the per-example ANS position for evaluation."""
        target_hidden = self.target_hidden_states(input_tuples, target_ids)
        batch_indices = torch.arange(
            target_hidden.shape[0],
            device=target_hidden.device,
        )
        answer_hidden = target_hidden[batch_indices, answer_positions]
        return self.head(answer_hidden)

    def forward(
        self,
        input_tuples: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return full target-position logits for compatibility/debugging."""
        return self.head(self.target_hidden_states(input_tuples, target_ids))
