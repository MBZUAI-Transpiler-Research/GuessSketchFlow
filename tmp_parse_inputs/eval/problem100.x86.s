	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3                               ## -- Begin function func0
LCPI0_0:
	.quad	0x3fe0000000000000              ## double 0.5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_func0
	.p2align	4, 0x90
_func0:                                 ## @func0
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_atof
	movsd	%xmm0, -16(%rbp)
	xorps	%xmm0, %xmm0
	ucomisd	-16(%rbp), %xmm0
	jbe	LBB0_2
## %bb.1:
	movsd	-16(%rbp), %xmm0                ## xmm0 = mem[0],zero
	movsd	LCPI0_0(%rip), %xmm1            ## xmm1 = mem[0],zero
	subsd	%xmm1, %xmm0
	roundsd	$10, %xmm0, %xmm0
	movsd	%xmm0, -24(%rbp)                ## 8-byte Spill
	jmp	LBB0_3
LBB0_2:
	movsd	LCPI0_0(%rip), %xmm0            ## xmm0 = mem[0],zero
	addsd	-16(%rbp), %xmm0
	roundsd	$9, %xmm0, %xmm0
	movsd	%xmm0, -24(%rbp)                ## 8-byte Spill
LBB0_3:
	movsd	-24(%rbp), %xmm0                ## 8-byte Reload
                                        ## xmm0 = mem[0],zero
	cvttsd2si	%xmm0, %eax
	addq	$32, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
.subsections_via_symbols
