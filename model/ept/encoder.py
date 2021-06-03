from typing import Tuple, List

import torch

from common.model.const import DEF_ENCODER
from common.model.types import Encoded, Text
from .chkpt import CheckpointingModule


def _gather_number_vectors(hidden: torch.Tensor, mask: torch.Tensor) -> List[Encoded]:
    # Compute the maximum number of indicated positions in the text
    batch_size, seq_len, hidden_size = hidden.shape

    batched_items = []
    for b in range(batch_size):
        row_items = []
        row_number_max = mask[b].max().item()
        for n in range(row_number_max + 1):  # Including the last number
            indices = mask[b].eq(n).nonzero(as_tuple=False).view(-1).tolist()
            assert len(indices) > 0

            begin = min(indices)
            end = max(indices) + 1

            # Copy masked positions. Shape [T, H].
            row_items.append(Encoded(hidden[b, begin:end], None))

        # Add batched vectors. Shape [N, T, H].
        if len(row_items):
            batched_items.append(Encoded.build_batch(*row_items))
        else:
            batched_items.append(Encoded.empty(0, 0, hidden_size, device=hidden.device))

    # Return B-list of [N, T, H].
    return batched_items


class TextEncoder(CheckpointingModule):
    """
    Model for encoding text.
    """

    def __init__(self, encoder: str = DEF_ENCODER):
        """
        Initiate Text Model instance.

        :param ModelConfig config: Model configuration instance
        """
        super().__init__(encoder=encoder)
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(encoder)
        self.pad_id = AutoTokenizer.from_pretrained(encoder).pad_token_id

    def forward(self, text: Text) -> Tuple[Encoded, List[Encoded]]:
        with torch.no_grad():
            # Find the last non-pad position
            text_length = text.sequence_lengths.max().item()
            # Cut off padded items to reduce GPU usage
            text: Text = text[:, :text_length]

        # Encode text
        # Replace PAD_ID (-1) with pad of tokenizer
        model_out = self.model(input_ids=text.tokens.pad_fill(self.pad_id),
                               attention_mask=text.attn_mask_float)[0]

        # Form an encoded output
        encoded: Encoded = Encoded(model_out, text.pad)

        # Gather numbers
        number_out = _gather_number_vectors(encoded.vector, text.numbers.indices)

        return encoded, number_out


__all__ = ['TextEncoder']
