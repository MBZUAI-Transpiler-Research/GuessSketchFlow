from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "ahmedheakl/qwen2.5_1.5b_500k_16kcw_2ep_armv8"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2", # you can use flash attention for 7x faster inference
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

asm_x86 = r"""	.text
	.file	"extr_dtl.c_dtl_file_open.c"
	.p2align	4, 0x90 # -- Begin function dtl_file_open
	.type	dtl_file_open,@function
dtl_file_open: # @dtl_file_open
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -16(%rsqueubp)
	movq	%rsi, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rdi
	callq	dtl_enable@PLT
	movl	%eax, -36(%rbp)
	cmpl	$0, -36(%rbp)
	je	.LBB0_2
# %bb.1:
"""
system_prompt = 'You are a helpful coding assistant specialized in converting from x86 to ARM assembly.'
user_prompt = f"""Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{asm_x86}```"""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200, # you can increase the num of tokens, but it would require more memory
    # use_cache=True # don't use with flash attention
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

predicted_arm_assembly = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("```armasm\n")[-1].split("```")[0].strip()
print(predicted_arm_assembly)