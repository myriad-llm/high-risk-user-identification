from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch


class CallRecordsDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(
        self, examples: List[Union[List[int], Tensor, Dict[str, Tensor]]]
    ) -> Dict[str, Tensor]:
        batch = _torch_collate_batch(examples, self.tokenizer)
        sz = batch.shape
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def mask_tokens(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class CallRecordsDataCollatorForClassification(
    CallRecordsDataCollatorForLanguageModeling
):
    def __call__(
        self, examples: List[Union[List[int], Tensor, Dict[str, Tensor]]]
    ) -> Dict[str, Tensor]:
        example = [t[0] for t in examples]
        label_example = [t[1] for t in examples] if examples[0][1] is not None else None
        msisdns = [t[2] for t in examples]

        batch = _torch_collate_batch(example, self.tokenizer)
        label_batch = (
            _torch_collate_batch(label_example, self.tokenizer)
            if label_example is not None
            else None
        )

        return {"input_ids": batch, "labels": label_batch, "msisdns": msisdns}
