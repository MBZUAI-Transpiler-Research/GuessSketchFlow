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
	ldr	x8, [sp, #16]
	subs	x8, x8, #2
	cset	w8, ge
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	mov	w8, #0
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	strb	w8, [sp, #31]
	b	LBB0_9
LBB0_2:
	mov	x8, #2
	str	x8, [sp, #8]
	b	LBB0_3
LBB0_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #8]
	ldr	x9, [sp, #8]
	mul	x8, x8, x9
	ldr	x9, [sp, #16]
	subs	x8, x8, x9
	cset	w8, gt
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	x8, [sp, #16]
	ldr	x10, [sp, #8]
	sdiv	x9, x8, x10
	mul	x9, x9, x10
	subs	x8, x8, x9
	subs	x8, x8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:
	mov	w8, #0
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	strb	w8, [sp, #31]
	b	LBB0_9
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=1
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	x8, [sp, #8]
	add	x8, x8, #1
	str	x8, [sp, #8]
	b	LBB0_3
LBB0_8:
	mov	w8, #1
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	strb	w8, [sp, #31]
	b	LBB0_9
LBB0_9:
	ldrb	w8, [sp, #31]
	and	w0, w8, #0x1
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

