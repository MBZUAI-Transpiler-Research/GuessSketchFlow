	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, #40]
	str	w1, [sp, #36]
	str	x2, [sp, #24]
	mov	w8, #2139095039
	fmov	s0, w8
	str	s0, [sp, #20]
	ldr	x8, [sp, #40]
	ldr	s0, [x8]
	ldr	x8, [sp, #24]
	str	s0, [x8]
	ldr	x8, [sp, #40]
	ldr	s0, [x8, #4]
	ldr	x8, [sp, #24]
	str	s0, [x8, #4]
	str	wzr, [sp, #16]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #36]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_10
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #36]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #16]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #12]
	ldr	s1, [x8, x9, lsl #2]
	fsub	s0, s0, s1
	fcvt	d0, s0
	fabs	d0, d0
	fcvt	s0, d0
	str	s0, [sp, #8]
	ldr	s0, [sp, #8]
	ldr	s1, [sp, #20]
	fcmp	s0, s1
	cset	w8, pl
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	s0, [sp, #8]
	str	s0, [sp, #20]
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #16]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #24]
	str	s0, [x8]
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #12]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #24]
	str	s0, [x8, #4]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #18]
	add	w8, w8, #1
	str	w8, [sp, #18]
	b	LBB0_1
LBB0_10:
	ldr	x8, [sp, #24]
	ldr	s0, [x8]
	ldr	x8, [sp, #24]
	ldr	s1, [x8, #4]
	fcmp	s0, s1
	cset	w8, le
	tbnz	w8, #0, LBB0_12
	b	LBB0_11
LBB0_11:
	ldr	x8, [sp, #24]
	ldr	s0, [x8]
	str	s0, [sp, #4]
	ldr	x8, [sp, #24]
	ldr	s0, [x8, #4]
	ldr	x8, [sp, #24]
	str	s0, [x8]
	ldr	s0, [sp, #4]
	ldr	x8, [sp, #24]
	str	s0, [x8, #4]
	b	LBB0_12
LBB0_12:
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

