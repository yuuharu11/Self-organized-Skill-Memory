import logging
import os

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download


class PaligemmaTokenizer:

    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        tokenizer_path = os.environ.get(
            "SKILLNET_PALIGEMMA_TOKENIZER",
            "gs://big_vision/paligemma_tokenizer.model",
        )
        path = download.maybe_download(tokenizer_path, gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    def tokenize_high_level_prefix(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the exact prefix used by the subtask decoder."""
        text = f"Task: {prompt.strip().replace('_', ' ').replace(chr(10), ' ')}. Subtask: "
        tokens = self._tokenizer.encode(text, add_bos=True)
        if len(tokens) > self._max_len:
            raise ValueError(f"Subtask prefix has {len(tokens)} tokens; max_len={self._max_len}")
        mask = [True] * len(tokens) + [False] * (self._max_len - len(tokens))
        tokens += [0] * (self._max_len - len(tokens))
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=np.bool_)

    def detokenize(self, tokens: np.ndarray | list[int]) -> str:
        """Decode a generated subtask, stopping before its EOS token."""
        ids = [int(token) for token in np.asarray(tokens).reshape(-1).tolist()]
        if 1 in ids:
            ids = ids[: ids.index(1)]
        else:
            # In a max-length truncation there is no EOS; ignore only trailing
            # padding, preserving any zero token in the interior.
            while ids and ids[-1] == 0:
                ids.pop()
        return self._tokenizer.decode(ids).strip()

    def tokenize_high_low_prompt(
        self, high_prompt: str, low_prompt: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize Task + Subtask target and mark only the subtask/EOS for CE."""
        prefix = f"Task: {high_prompt.strip().replace('_', ' ').replace(chr(10), ' ')}. Subtask: "
        prefix_ids = self._tokenizer.encode(prefix, add_bos=True)
        target_ids = self._tokenizer.encode(low_prompt.strip(), add_bos=False)
        # PaliGemma uses token 1 for EOS. Include it in the target so decoding has
        # an explicit, learned terminator.
        ids = prefix_ids + target_ids + [1]
        if len(ids) > self._max_len:
            raise ValueError(f"Subtask sequence has {len(ids)} tokens; max_len={self._max_len}")
        length = len(ids)
        tokens = ids + [0] * (self._max_len - length)
        mask = [True] * length + [False] * (self._max_len - length)
        ar_mask = [False] * len(prefix_ids) + [True] * (length - len(prefix_ids)) + [False] * (self._max_len - length)
        loss_mask = [False] * len(prefix_ids) + [True] * (length - len(prefix_ids)) + [False] * (self._max_len - length)
        return tuple(np.asarray(value, dtype=dtype) for value, dtype in (
            (tokens, np.int32), (mask, np.bool_), (ar_mask, np.int32), (loss_mask, np.bool_)
        ))


class FASTTokenizer:

    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(self, prompt: str, state: np.ndarray,
                 actions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (self._paligemma_tokenizer.encode("Action: ") + action_tokens_in_pg.tolist() +
                              self._paligemma_tokenizer.encode("|"))
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently.")
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode([action_tokens.tolist()], time_horizon=action_horizon,
                                           action_dim=action_dim)[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
