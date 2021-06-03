import re
from typing import List, Union, Tuple

from common.const.operand import PREFIX_LEN
from common.const.pad import PAD_ID
from common.seed.pattern import NUMBER_OR_FRACTION_PATTERN as NOF_WITH_START_PATTERN
from common.model.base import *
from .label import Label


# Remove '^' from the pattern.
NUMBER_OR_FRACTION_PATTERN = re.compile(NOF_WITH_START_PATTERN.pattern[1:])


def _number_index_reader(token: int) -> str:
    if token == PAD_ID:
        return '__'
    else:
        return '%2d' % token


def _add_space_around_number(text: str) -> Tuple[str, dict]:
    orig_to_spaced = {}
    spaced = []
    for orig, token in enumerate(text.split()):
        spaced_tokens = re.sub('\\s+', ' ', NUMBER_OR_FRACTION_PATTERN.sub(' \\1 ', token)).strip().split()
        orig_to_spaced[orig] = len(spaced)
        token_reset = False

        for new in spaced_tokens:
            if not token_reset and NUMBER_OR_FRACTION_PATTERN.fullmatch(new):
                orig_to_spaced[orig] = len(spaced)
                token_reset = True
            # Add token
            spaced.append(new)

    return ' '.join(spaced), orig_to_spaced


def _remove_special_prefix(token: str) -> str:
    # Handle different kind of spacing prefixes...
    from transformers import SPIECE_UNDERLINE
    if token == SPIECE_UNDERLINE:
        return ' '
    if token.startswith(SPIECE_UNDERLINE):
        return token[len(SPIECE_UNDERLINE):]
    if token.startswith('##'):
        return token[2:]
    return token


class Text(TypeTensorBatchable, TypeSelectable):
    #: Tokenized raw text (tokens are separated by whitespaces)
    raw: Union[str, List[str]]
    #: Tokenized text
    tokens: Label
    #: Number index label for each token
    numbers: Label
    #: Number snippets for each number label
    snippets: Union[Label, List[Label]]

    def __init__(self, raw: Union[str, List[str]], tokens: Label, numbers: Label, snippets: Union[Label, List[Label]]):
        super().__init__()
        self.raw = raw
        self.tokens = tokens
        self.numbers = numbers
        self.snippets = snippets

    def __getitem__(self, item) -> 'Text':
        if type(item) is int and self.is_batched:
            return Text(raw=self.raw[item], tokens=self.tokens[item], numbers=self.numbers[item],
                        snippets=self.snippets[item])
        else:
            return super().__getitem__(item)

    @property
    def shape(self) -> torch.Size:
        return self.tokens.shape

    @property
    def pad(self) -> torch.BoolTensor:
        return self.tokens.pad

    @property
    def attn_mask_float(self) -> torch.Tensor:
        return self.tokens.attn_mask_float

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.tokens.sequence_lengths

    @classmethod
    def build_batch(cls, *items: 'Text') -> 'Text':
        return Text(raw=[item.raw for item in items],
                    tokens=Label.build_batch(*[item.tokens for item in items]),
                    numbers=Label.build_batch(*[item.numbers for item in items]),
                    snippets=[item.snippets for item in items])

    @classmethod
    def from_dict(cls, raw: dict, tokenizer, number_window: int) -> 'Text':
        # Tokenize the text
        spaced, orig_to_new_wid = _add_space_around_number(raw['text'])
        tokens: List[int] = tokenizer.encode(spaced)
        text: str = ' '.join(tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True))

        # Read numbers
        numbers = raw['numbers']
        wid_to_nid = {orig_to_new_wid[token_id]: int(number['key'][PREFIX_LEN:])
                      for number in numbers
                      for token_id in number['tokenRange']}

        # Find position of numbers
        token_nids = []
        current_nid = PAD_ID
        current_wid = 0
        string_left = ' ' + spaced.lower()
        for token in tokenizer.convert_ids_to_tokens(tokens):
            if token in tokenizer.all_special_tokens:
                current_nid = PAD_ID
            else:
                # Find whether this is the beginning of the word.
                # We don't use SPIECE_UNDERLINE or ## because ELECTRA separates comma or decimal point...
                if string_left[0].isspace():
                    current_nid = wid_to_nid.get(current_wid, PAD_ID)
                    current_wid += 1
                    string_left = string_left[1:]

                token_string = _remove_special_prefix(token)
                assert string_left.startswith(token_string)
                string_left = string_left[len(token_string):]

            token_nids.append(current_nid)

        # Set pad token id as PAD_ID (This will be replaced inside a model instance)
        tokens = [tok if tok != tokenizer.pad_token_id else PAD_ID
                  for tok in tokens]
        assert len(tokens) == len(token_nids)

        # Make snippet of numbers
        number_snippets = []
        for nid in range(max(token_nids) + 1):
            token_start = token_nids.index(nid)
            window_start = token_start - number_window
            window_end = token_start + number_window + 1

            pad_front = [PAD_ID] * max(0, -window_start)
            pad_back = [PAD_ID] * max(0, window_end - len(tokens))
            snippet_nid = tokens[max(window_start, 0):window_end]

            number_snippets.append(pad_front + snippet_nid + pad_back)
            assert len(number_snippets[-1]) == number_window * 2 + 1

        assert len(number_snippets) == max(token_nids) + 1
        return Text(raw=text, tokens=Label.from_list(tokens), numbers=Label.from_list(token_nids),
                    snippets=Label.from_list(number_snippets))

    def as_dict(self) -> dict:
        return dict(raw=self.raw, tokens=self.tokens, numbers=self.numbers, snippets=self.snippets)

    def to_human_readable(self, tokenizer=None) -> dict:
        if tokenizer is None:
            text_converter = None
        else:
            text_converter = lambda t: tokenizer.convert_ids_to_tokens(t) if t != PAD_ID else ''

        return {
            'raw': human_readable_form(self.raw),
            'tokens': self.tokens.to_human_readable(converter=text_converter),
            'numbers': self.numbers.to_human_readable(converter=_number_index_reader),
            'snippets': human_readable_form(self.snippets, converter=text_converter)
        }
