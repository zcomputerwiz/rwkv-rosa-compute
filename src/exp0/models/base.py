"""Base input seam for models accepting Match-3 multi-hot features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedWrapper(nn.Module):
    """Project tuple and continuation features through one shared input layer.

    ``vocab_size`` is the number of task token IDs that can be fed back as
    continuation inputs. ``output_vocab_size`` may be larger: the authors'
    positive-control Llama keeps its 32k LM head even though Match-3 labels use
    only a small subset of those output classes. Decoupling the two dimensions
    reproduces that loss geometry without inventing thousands of fake input
    features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_input: int,
        hidden_size: int,
        vocab_size: int,
        *,
        output_vocab_size: int | None = None,
        target_feature_indices: torch.Tensor | None = None,
        input_feature_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_input = d_input
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_vocab_size = (
            vocab_size if output_vocab_size is None else output_vocab_size
        )
        if self.output_vocab_size < vocab_size:
            raise ValueError(
                "output_vocab_size must cover every task vocabulary ID: "
                f"got output={self.output_vocab_size}, task={vocab_size}."
            )

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
                    "target_feature_indices must contain one entry per task vocab id."
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
        self.head = nn.Linear(hidden_size, self.output_vocab_size, bias=False)

    def _tuple_hidden(self, input_tuples: torch.Tensor) -> torch.Tensor:
        weight = self.input_proj.weight[:, : self.d_input]
        return F.linear(input_tuples, weight, self.input_proj.bias)

    def _target_hidden(self, target_ids: torch.Tensor) -> torch.Tensor:
        mapped = self.target_feature_indices[target_ids]
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
