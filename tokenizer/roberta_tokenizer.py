"""
Minimal RoBERTa tokenizer wrapper using Hugging Face `tokenizers`.

Requires vocab.json + merges.txt.
"""

from enum import Enum
from collections import UserDict
from typing import List, Optional, Union

import numpy as np

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing


class PaddingStrategy(Enum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class BatchEncoding(UserDict):
    """
    Holds the output of Tokenizer.
    """

    def __init__(self, data=None, tensor_type=None, prepend_batch_axis=False):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    def convert_to_tensors(self, tensor_type=None, prepend_batch_axis=False):
        if tensor_type is None:
            return self

        if tensor_type == "pt":
            import torch
            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        elif tensor_type == "jt":
            import jittor
            as_tensor = jittor.Var
            is_tensor = jittor.is_var
        else:
            def _is_numpy(x):
                return isinstance(x, np.ndarray)

            as_tensor = np.asarray
            is_tensor = _is_numpy

        for key, value in self.items():
            if prepend_batch_axis:
                value = [value]
            if not is_tensor(value):
                self[key] = as_tensor(value)
        return self


class RobertaTokenizer:
    """
    Byte-level BPE tokenizer for RoBERTa.
    """

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        add_prefix_space: bool = True,
        model_max_length: int = 512,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        cls_token: str = "<s>",
        sep_token: str = "</s>",
        mask_token: str = "<mask>",
    ):
        self.tokenizer = ByteLevelBPETokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            add_prefix_space=add_prefix_space,
        )
        self.model_max_length = model_max_length
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

        self._init_special_tokens()

    def _init_special_tokens(self):
        vocab = self.tokenizer.get_vocab()
        self.unk_token_id = vocab.get(self.unk_token)
        self.pad_token_id = vocab.get(self.pad_token)
        self.cls_token_id = vocab.get(self.cls_token)
        self.sep_token_id = vocab.get(self.sep_token)
        self.mask_token_id = vocab.get(self.mask_token)

        if self.cls_token_id is None or self.sep_token_id is None:
            raise ValueError("Missing <s> or </s> in vocab.")

        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.cls_token} $A {self.sep_token}",
            pair=f"{self.cls_token} $A {self.sep_token}{self.sep_token} $B {self.sep_token}",
            special_tokens=[
                (self.cls_token, self.cls_token_id),
                (self.sep_token, self.sep_token_id),
            ],
        )
        self.tokenizer.enable_padding(
            direction="right",
            pad_id=self.pad_token_id if self.pad_token_id is not None else 1,
            pad_token=self.pad_token,
        )

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
        **kwargs,
    ):
        is_batched = isinstance(text, (list, tuple))
        if max_length is None:
            max_length = self.model_max_length

        if is_batched:
            batch_texts = text
            batch_pairs = text_pair if isinstance(text_pair, (list, tuple)) else None
            encodings = self.tokenizer.encode_batch(batch_texts, batch_pairs, add_special_tokens=add_special_tokens)
            return self._batch_format(
                encodings,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
            )

        encoding = self.tokenizer.encode(text, text_pair, add_special_tokens=add_special_tokens)
        return self._format(
            encoding,
            padding=padding,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            prepend_batch_axis=True,
        )

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text, add_special_tokens=False).tokens

    def get_input_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _format(
        self,
        encoding,
        padding,
        max_length,
        return_tensors,
        return_attention_mask,
        return_token_type_ids,
        prepend_batch_axis=False,
    ):
        data = {"input_ids": encoding.ids}
        if return_attention_mask:
            data["attention_mask"] = encoding.attention_mask
        if return_token_type_ids:
            data["token_type_ids"] = encoding.type_ids

        if padding:
            data = self._pad(data, max_length=max_length)

        return BatchEncoding(data, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)

    def _batch_format(
        self,
        encodings,
        padding,
        max_length,
        return_tensors,
        return_attention_mask,
        return_token_type_ids,
    ):
        batch = {"input_ids": [e.ids for e in encodings]}
        if return_attention_mask:
            batch["attention_mask"] = [e.attention_mask for e in encodings]
        if return_token_type_ids:
            batch["token_type_ids"] = [e.type_ids for e in encodings]

        if padding:
            batch = self._batch_pad(batch, max_length=max_length, padding=padding)

        return BatchEncoding(batch, tensor_type=return_tensors)

    def _batch_pad(self, batch, max_length=None, padding=True):
        padding_strategy, max_length = self._get_padding_strategies(padding, max_length)
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(ids) for ids in batch["input_ids"])
        for i in range(len(batch["input_ids"])):
            item = {k: v[i] for k, v in batch.items()}
            padded = self._pad(item, max_length=max_length)
            for key, value in padded.items():
                batch[key][i] = value
        return batch

    def _pad(self, data, max_length):
        input_ids = data["input_ids"]
        if max_length is None:
            max_length = len(input_ids)
        if len(input_ids) >= max_length:
            return data
        pad_len = max_length - len(input_ids)
        data["input_ids"] = input_ids + [self.pad_token_id] * pad_len
        if "attention_mask" in data:
            data["attention_mask"] = data["attention_mask"] + [0] * pad_len
        if "token_type_ids" in data:
            data["token_type_ids"] = data["token_type_ids"] + [0] * pad_len
        return data

    def _get_padding_strategies(self, padding=False, max_length=None):
        padding_strategy = PaddingStrategy.DO_NOT_PAD
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            else:
                padding_strategy = padding
        if max_length is None:
            max_length = self.model_max_length
        return padding_strategy, max_length
