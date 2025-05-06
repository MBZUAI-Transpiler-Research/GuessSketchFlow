	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	w0, [sp, #28]
	str	w1, [sp, #24]
	str	w2, [sp, #20]
	str	x3, [sp, #8]
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #20]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	ldr	w8, [sp, #28]
	ldr	w9, [sp, #20]
	add	w8, w8, w9
	ldr	x9, [sp, #8]
	str	w8, [x9]
	ldr	x8, [sp, #8]
	str	wzr, [x8, #4]
	b	LBB0_3
LBB0_2:
	ldr	w8, [sp, #28]
	ldr	w9, [sp, #24]
	add	w8, w8, w9
	ldr	x9, [sp, #8]
	str	w8, [x9]
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	ldr	x9, [sp, #8]
	str	w8, [x9, #4]
	b	LBB0_3
LBB0_3:
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

