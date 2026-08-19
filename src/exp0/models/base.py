"""Base InputEmbedWrapper seam for models accepting multi-hot input vectors."""

import torch
import torch.nn as nn


class InputEmbedWrapper(nn.Module):
    """Project synthetic task inputs into a sequence-model backbone."""

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

        self.tuple_proj = nn.Linear(d_input, hidden_size)
        self.target_embed = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=pad_token_id,
        )
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def target_hidden_states(
        self,
        input_tuples: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return backbone hidden states aligned with target-token positions."""
        _, n, _ = input_tuples.shape

        tuple_embeds = self.tuple_proj(input_tuples)
        target_embeds = self.target_embed(target_ids)
        inputs_embeds = torch.cat([tuple_embeds, target_embeds], dim=1)
        hidden_states = self.backbone(inputs_embeds=inputs_embeds)
        return hidden_states[:, n:, :]

    def loss_logits(
        self,
        input_tuples: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Project only positions that participate in next-token loss.

        The previous training path projected all target positions to vocabulary
        logits and then copied ``logits[:, :-1]`` into a contiguous tensor.
        Projecting ``hidden[:, :-1]`` directly removes the unused final logits
        and the second nearly-full logits allocation without changing the loss.
        """
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
