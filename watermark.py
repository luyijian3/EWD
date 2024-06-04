# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt
import pdb
import scipy.stats

import torch
from torch import Tensor
from transformers import LogitsProcessor

from nltk.util import ngrams

# from normalizers import normalization_strategy_lookup

class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485917,  # just a large prime number to create a rng seed with sufficient bit width
        select_green_tokens: bool = True,
        entropy_threshold: float = 0.0,
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.entropy_threshold = entropy_threshold
        alpha = torch.exp(torch.tensor(self.delta)).item()
        self.z_value = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))

    def _seed_rng(self, input_ids: torch.LongTensor, hash_key: int, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(hash_key * prev_token) ### newly change self.hash_key to hash_key ###
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        self._seed_rng(input_ids, self.hash_key)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        if self.select_green_tokens: # directly
            greenlist_ids = vocab_permutation[:greenlist_size] # new
        else: # select green via red
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]  # legacy behavior
        return greenlist_ids

    def calculate_entropy(self, model, tokenized_text) -> list[float]:
        """Calculate the entropy of the tokenized text using the model."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            denoms = 1+(self.z_value * probs)
            renormed_probs = probs / denoms
            sum_renormed_probs = renormed_probs.sum(dim=-1)
            entropy=sum_renormed_probs[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]

class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator()

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores

class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        tokenizer: None,
        ignore_repeated_bigrams: bool = False,
        type: str = 'wllm', # wllm, sweet, ewd
        model:None,
        acc:None,

        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        self.tokenizer = tokenizer
        self.rng = torch.Generator()
        self.type=type
        if model:
            self.model = model.to(acc.device)
            self.acc = acc

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams: 
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."


    def _compute_z_score(self, observed_count, w):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * torch.sum(w, dim=0)
        denom = sqrt(torch.sum(torch.square(w), dim=0) * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _score_sequence(
        self,
        input_ids: Tensor,
        prefix_len: int,
        entropy,# prefix is removed
    ):
        score_dict = dict()
        if self.ignore_repeated_bigrams:
            raise NotImplementedError("not used")
        else:
            prefix_len = max(self.min_prefix_len, prefix_len)

            num_tokens_scored = len(input_ids) - prefix_len
            if self.type in ['sweet','ewd']:
                assert len(entropy) == num_tokens_scored, 'entropy length should be the same as num_tokens_scored'
            if num_tokens_scored < 1:
                print(f"only {num_tokens_scored} scored : cannot score.")
                score_dict["invalid"] = True
                return score_dict

            green_token_count, flag_all= 0, []
            for idx in range(prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                flag_all.append(curr_token in greenlist_ids)
        if self.type=='wllm':
            score_dict.update(dict(z_score=self._compute_z_score(len([i for i in flag_all if i]), torch.tensor([1.0 for _ in range(num_tokens_scored)])).item()))
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
            score_dict.update(dict(num_green_tokens=len([i for i in flag_all if i])))
        elif self.type=='sweet':
            num_token_sweet=len([i for i in entropy if i>self.entropy_threshold])
            green = len([i for i,j in zip(entropy,flag_all) if i>self.entropy_threshold and j])
            score_dict.update(dict(z_score=self._compute_z_score(green, torch.tensor([1.0 for _ in range(num_token_sweet)])).item()))
            score_dict.update(dict(num_tokens_scored=num_token_sweet))
            score_dict.update(dict(num_green_tokens=green))
        elif self.type == 'ewd':
            SE=torch.sub(torch.tensor(entropy), torch.min(torch.tensor(entropy)))
            green_token=torch.stack([tensor for tensor, flag in zip(SE, flag_all) if flag])
            green_token_count = torch.sum(green_token, dim=0)
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, SE).item()))
            score_dict.update(dict(num_green_tokens=green_token_count.item()))
        score_dict.update(dict(entropy=entropy))
        score_dict.update(dict(flag=flag_all))
        return score_dict

    def detect(
        self,
        tokenized_text: torch.Tensor = None,
        tokenized_prefix: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        assert tokenized_text is not None, "Must pass either tokenized string"
        if self.type in ['sweet','ewd']:
            #calculate entropy
            entropy = self.calculate_entropy(self.model, tokenized_text.to(self.acc.device))
        else:
            entropy=[]
        score_dict = self._score_sequence(input_ids=tokenized_text, prefix_len=len(tokenized_prefix),entropy=entropy[len(tokenized_prefix):], **kwargs)
        return score_dict

