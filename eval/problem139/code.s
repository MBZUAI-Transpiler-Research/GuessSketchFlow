	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	w0, [sp, #8]
	ldr	w8, [sp, #8]
	mov	w10, #2
	sdiv	w9, w8, w11
	mul	w9, w9, w11
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_3
	b	LBB0_1
LBB0_1:
	ldr	w8, [sp, #8]
	subs	w8, w8, #8
	cset	w8, lt
	tbnz	w8, #0, LBB0_3
	b	LBB0_2
LBB0_2:
	mov	w8, #1
	str	w8, [sp, #12]
	b	LBB0_4
LBB0_3:
	str	wzr, [sp, #12]
	b	LBB0_4
LBB0_4:
	ldr	w0, [sp, #12]
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

