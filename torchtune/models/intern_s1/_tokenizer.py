from typing import Optional

from transformers import AutoTokenizer

from torchtune.data import PromptTemplate
from torchtune.models.qwen3._tokenizer import (
    DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
    ENDOFTEXT,
    IM_END,
    QWEN3_SPECIAL_TOKENS,
    Qwen3Tokenizer,
)


class InternS1Tokenizer(Qwen3Tokenizer):  # noqa: N801
    def __init__(
        self,
        path: str,
        merges_file: str,
        special_tokens: dict[str, int] = QWEN3_SPECIAL_TOKENS,
        max_seq_len: Optional[int] = None,
        *,
        prompt_template: Optional[PromptTemplate] = None,
        errors: str = "replace",
        unk_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: str = IM_END,
        pad_token: Optional[str] = ENDOFTEXT,
        bpe_cache_size: int = DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
        truncation_type: str = "right",
    ):
        super().__init__(
            path=path,
            merges_file=merges_file,
            special_tokens=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            bpe_cache_size=bpe_cache_size,
        )

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            "haydn-jones/Intern-S1-mini-Qwen3-8B",
            trust_remote_code=True,
        )
        print("!" * 80)
        print("USING PATCHED INTERN S1 TOKENIZER THAT PROPERLY HANDLES <FASTA>")
        print("USING PATCHED INTERN S1 TOKENIZER THAT PROPERLY HANDLES <FASTA>")
        print("USING PATCHED INTERN S1 TOKENIZER THAT PROPERLY HANDLES <FASTA>")
        print("!" * 80)

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        tokens = self.hf_tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        token_ids: list[int] = []
        if add_bos and self.bos_id is not None:
            token_ids.append(self.bos_id)

        token_ids.extend(tokens)

        if add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids
