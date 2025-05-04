"""Implementation of BART-based sequence-to-sequence model for assembly transpilation."""

import gc
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

from src.helpers.dataset import DatasetInstance
from src.helpers.model import Model, InferenceConfig, PredictionResult


class BartLargeModel(Model):
    """Implementation of BART-based transpiler model.

    This class implements the Model interface using the BART architecture,
    which is well-suited for sequence-to-sequence tasks like assembly
    code translation. BART uses a bidirectional encoder and autoregressive
    decoder architecture.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize BART model and tokenizer.

        Args:
            model_name: Hugging Face model ID or local checkpoint path
            device: Device to place model on (cuda/cpu)
        """
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        tokenizer.model_max_length = 2048

        config = AutoConfig.from_pretrained("facebook/bart-large")
        config.vocab_size = len(tokenizer)
        config.max_position_embeddings = 2048

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            from_tf=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            config=config
        ).to(device)

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(tokenizer))

        if self.model.config.decoder_start_token_id is None:
            print(
                f"config.decoder_start_token_id is set to None, so auto setting to to BOS"
            )
            self.model.config.decoder_start_token_id = tokenizer.bos_token_id

        super().__init__(tokenizer, device=device)
        self.model.eval()

    def prepare_prompt(
        self,
        source_code: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Prepare text prompt for BART model.

        Unlike chat-based models, BART uses a simple cloze-style input format.

        Args:
            source_code: Source assembly code
            source_lang: Source assembly language
            target_lang: Target assembly language

        Returns:
            Formatted prompt string
        """
        return source_code

    def tokenize(
        self,
        text: Union[str, List[str]],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input text and move tensors to device.

        Args:
            text: Input text or list of texts to tokenize

        Returns:
            Dict containing tokenized tensors on target device
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

    def decode(
        self,
        token_ids: torch.Tensor,
    ) -> Union[str, List[str]]:
        """Decode token IDs back to text.

        Args:
            token_ids: Tensor of token IDs to decode
            **kwargs: Additional decoding arguments

        Returns:
            Single string if batch size is 1, otherwise list of strings
        """
        decoded = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
        )
        if token_ids.shape[0] == 1:
            return decoded[0]
        return decoded

    def infer(
        self,
        input_tokens: Dict[str, torch.Tensor],
        config: Optional[InferenceConfig] = None,
    ):
        """Perform inference using BART seq2seq model.

        Args:
            input_tokens: Dict containing input_ids and attention_mask
            config: Optional inference configuration
            **kwargs: Additional generation arguments

        Returns:
            InferenceResult containing generated tokens and metadata
        """
        if config is None:
            config = InferenceConfig()

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=config.max_length,
            temperature=config.temperature,
            num_return_sequences=config.num_return_sequences,
            num_beams=config.beam_size,
            do_sample=(config.temperature > 0),
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
            no_repeat_ngram_size=0,
        )
        alignments = self.get_alignments(outputs)
        confidence = self.get_confidence(outputs)

        return (input_tokens.input_ids, outputs.sequences, alignments, confidence)

    def extract(self, program):
        sections = {}
        cloze = ""
        in_chunk = False
        for l in [l+"\n" for l in program.split("\n")]:
            l2 = l.strip()
            # if it's a .LC# line or a global function name label line
            # if l2 and l2[0] not in {'.', '_'} and '.' not in l2 and l2[-1] == ":":
            if l2 and l2[0] not in {'.'} and '.' not in l2 and l2[-1] == ":":
                in_chunk = True
                section = l2[:-1]
                cloze += f"{{{section}}}"
                sections[section] = ""
            if in_chunk:
                sections[section] += l
                if l2.strip() in {".cfi_endproc", "jr	ra"}:
                    in_chunk = False
            else:
                cloze += l
        return sections, cloze

    def get_alignments(self, pred_outputs, top_k=10):
        attentions = [
            xattn[-1].mean(dim=1)[:, 0]  # (output_len, input_len)
            for xattn in pred_outputs.cross_attentions
        ]

        out_seq_len = pred_outputs.sequences.shape[-1]
        aligned_tokens = []

        for out_idx in range(out_seq_len):
            if out_idx >= len(attentions):
                break

            # Get attention weights for this output token
            alignment = attentions[out_idx][0]  # (input_len,)

            # Get top-k input indices by attention
            top_indices = alignment.topk(top_k).indices.tolist()

            aligned_tokens.append(top_indices)

        return aligned_tokens

    def get_confidence(self, outputs):
        confidences = []
        scores = outputs.scores
        generated_tokens = outputs.sequences[:, -len(scores):]

        for step, (logits, tokens) in enumerate(zip(scores, generated_tokens.T)):
            probs = F.softmax(logits, dim=-1)
            batch_conf = probs[range(probs.size(0)), tokens]
            confidences.extend(batch_conf.tolist())
        
        return confidences

    def predict(
        self,
        instance: DatasetInstance,
        config: Optional[InferenceConfig] = None,
    ) -> PredictionResult:
        sections, cloze = self.extract(instance.source)
        sections_pred = {}

        self.model.eval()

        with torch.no_grad():
            for section_name, section_code in sections.items():
                tokenized_input = self.tokenize(section_code)

                sections_pred[section_name] = self.infer(
                    tokenized_input, config
                )
                gc.collect()

        tokenized_cloze = self.tokenize(cloze)
        cloze_pred = self.infer(tokenized_cloze, config)

        source_offset = 0

        source_tokens = torch.as_tensor(
            [[]], dtype=torch.int64
        ).to(self.device)
        pred_tokens = torch.as_tensor(
            [[]], dtype=torch.int64
        ).to(self.device)
        merged_confidence = []
        merged_alignments = []

        def merge_sections(section):
            nonlocal source_tokens, pred_tokens, source_offset, merged_alignments, merged_confidence

            (source_ids, pred_ids, alignments, confidence) = section

            source_tokens = torch.cat((source_tokens, source_ids), dim=1)
            pred_tokens = torch.cat((pred_tokens, pred_ids), dim=1)
            merged_confidence.extend(confidence)

            merged_alignments.extend(
                [
                    [src_idx + source_offset for src_idx in src_idxes]
                    for src_idxes in alignments
                ]
            )

            source_offset += len(source_ids[0])

            while len(merged_alignments) < len(pred_tokens[0]):
                merged_alignments.append([])

        for section_name, section in sections_pred.items():
            merge_sections(section)

        merge_sections(cloze_pred)

        return PredictionResult(
            instance_id=instance.instance_id,
            source=source_tokens,
            pred=pred_tokens,
            alignments=merged_alignments,
            confidence=merged_confidence,
        )
