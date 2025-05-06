	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, #32]
	str	w1, [sp, #28]
	str	x2, [sp, #16]
	mov	w8, #2147483647
	str	w8, [sp, #12]
	mov	w8, #-1
	str	w8, [sp, #8]
	str	wzr, [sp, #4]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #4]
	ldr	w9, [sp, #28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #4]
	ldr	w8, [x8, x9, lsl #2]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #4]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, lt
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]
	adds	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #4]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #12]
	ldr	w8, [sp, #4]
	str	w8, [sp, #8]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_1
LBB0_8:
	ldr	w8, [sp, #8]
	adds	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
                                        ; kill: def $x8 killed $xzr
	str	xzr, [sp, #40]
	b	LBB0_11
LBB0_10:
	ldr	w8, [sp, #12]
	ldr	x9, [sp, #16]
	str	w8, [x9]
	ldr	w8, [sp, #8]
	ldr	x9, [sp, #16]
	str	w8, [x9, #4]
	ldr	x8, [sp, #16]
	str	x8, [sp, #40]
	b	LBB0_11
LBB0_11:
	ldr	x0, [sp, #40]
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

