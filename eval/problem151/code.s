	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	w0, [sp, #24]
	str	w1, [sp, #20]
	str	w2, [sp, #16]
	mov	w8, #1
	str	w8, [sp, #12]
	ldr	w8, [sp, #24]
	subs	w8, w8, #2
	cset	w8, ge
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	str	wzr, [sp, #12]
	b	LBB0_2
LBB0_2:
	mov	w8, #2
	str	w8, [sp, #8]
	b	LBB0_3
LBB0_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #8]
	mul	w8, w8, w9
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #24]
	ldr	w10, [sp, #8]
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=1
	str	wzr, [sp, #12]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=1
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_3
LBB0_8:
	ldr	w8, [sp, #12]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_11
	b	LBB0_9
LBB0_9:
	ldr	w8, [sp, #20]
	str	w8, [sp, #28]
	b	LBB0_11
LBB0_10:
	ldr	w8, [sp, #16]
	str	w8, [sp, #28]
	b	LBB0_11
LBB0_11:
	ldr	w0, [sp, #28]
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

