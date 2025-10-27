from typing import List, Optional

from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType
from torchtune.models.intern_s1._component_builders import intern_s1, lora_intern_s1
from torchtune.models.intern_s1._tokenizer import (
    QWEN3_SPECIAL_TOKENS,
    InternS1Tokenizer,
)
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.transformer import TransformerDecoder
from torchtune.modules.transforms.tokenizers import parse_hf_tokenizer_json


def intern_s1_mini() -> TransformerDecoder:
    """
    Builder for creating a InternS1 mini model initialized w/ the default parameter values
    from https://huggingface.co/internlm/Intern-S1-mini

    Returns:
        TransformerDecoder: Instantiation of InternS1 mini model
    """
    return intern_s1(
        vocab_size=153216,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def lora_intern_s1_mini(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a InternS1 Mini instruct model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_7b_instruct`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Qwen3 8B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_intern_s1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=153216,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def intern_s1_tokenizer(
    path: str,
    merges_file: str,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
    truncation_type: str = "right",
    **kwargs,
) -> InternS1Tokenizer:
    """
    Tokenizer for Intern-S1.

    Args:
        path (str): path to the vocab.json file.
        merges_file (str): path to the merges.txt file.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen3 special tokens.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.
            Default is None.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Returns:
        Qwen3Tokenizer: Instantiation of the Qwen3 tokenizer
    """
    special_tokens = (
        QWEN3_SPECIAL_TOKENS
        if special_tokens_path is None
        else parse_hf_tokenizer_json(special_tokens_path)
    )

    if prompt_template is not None:
        prompt_template = _get_prompt_template(prompt_template)  # pyright: ignore[reportAssignmentType]

    return InternS1Tokenizer(
        path=path,
        merges_file=merges_file,
        special_tokens=special_tokens,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,  # pyright: ignore[reportArgumentType]
        truncation_type=truncation_type,
        **kwargs,
    )
