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
	str	w1, [sp, #4]
	ldr	w8, [sp, #4]
	ldr	w9, [sp, #8]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	mov	w8, #-1
	str	w8, [sp, #12]
	b	LBB0_8
LBB0_2:
	ldr	w8, [sp, #4]
	ldr	w9, [sp, #8]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_5
	b	LBB0_3
LBB0_3:
	ldr	w8, [sp, #4]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:
	mov	w8, #-1
	str	w8, [sp, #12]
	b	LBB0_8
LBB0_5:
	ldr	w8, [sp, #4]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:
	ldr	w8, [sp, #4]
	subs	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_8
LBB0_7:
	ldr	w8, [sp, #4]
	str	w8, [sp, #12]
	b	LBB0_8
LBB0_8:
	ldr	w0, [sp, #12]
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

