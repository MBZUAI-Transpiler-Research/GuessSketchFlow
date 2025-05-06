	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	x0, [sp, #16]
	str	w1, [sp, #12]
	str	wzr, [sp, #4]
	str	wzr, [sp, #8]
	mov	w8, #1
	str	w8, [sp]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #16]
	ldr	w9, [sp]
	subs	w9, w9, #1
	ldr	s1, [x8, w9, sxtw #2]
	fcmp	s0, s1
	cset	w8, le
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #8]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #16]
	ldr	w9, [sp]
	subs	w9, w9, #1
	ldr	s1, [x8, w9, sxtw #2]
	fcmp	s0, s1
	cset	w8, pl
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #4]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp]
	add	w8, w8, #1
	str	w8, [sp]
	b	LBB0_1
LBB0_8:
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #4]
	add	w8, w8, w9
	subs	w8, w8, #2
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	mov	w8, #0
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	strb	w8, [sp, #31]
	b	LBB0_11
LBB0_10:
	mov	w8, #1
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	strb	w8, [sp, #31]
	b	LBB0_11
LBB0_11:
	ldrb	w8, [sp, #31]
	and	w0, w8, #0x1
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

