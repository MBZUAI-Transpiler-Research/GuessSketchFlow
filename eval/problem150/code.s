	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #80
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	add	x29, sp, #64
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	stur	w1, [x29, #-12]
	stur	x2, [x29, #-24]
	ldur	x8, [x29, #-24]
	str	wzr, [x8]
	stur	wzr, [x29, #-28]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-28]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_6
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-28]
	ldr	x0, [x8, x9, lsl #3]
	bl	_strlen
	mov	x9, #2
	udiv	x8, x0, x9
	mul	x8, x8, x9
	subs	x8, x0, x8
	subs	x8, x8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-28]
	ldr	x8, [x8, x9, lsl #3]
	ldur	x9, [x29, #-8]
	ldur	x10, [x29, #-24]
	ldrsw	x11, [x11]
	str	x8, [x9, x11, lsl #3]
	ldur	x9, [x29, #-24]
	ldr	w8, [x9]
	add	w8, w8, #1
	str	w8, [x9]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_1
LBB0_6:
	stur	wzr, [x29, #-28]
	b	LBB0_7
LBB0_7:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_9 Depth 2
	ldur	w8, [x29, #-28]
	ldur	x9, [x29, #-24]
	ldr	w9, [x9]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_18
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_7 Depth=1
	stur	wzr, [x29, #-32]
	b	LBB0_9
LBB0_9:                                 ;   Parent Loop BB0_7 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldur	w8, [x29, #-32]
	ldur	x9, [x29, #-24]
	ldr	w9, [x9]
	ldur	w10, [x29, #-28]
	subs	w9, w9, w10
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_16
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_9 Depth=2
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-32]
	ldr	x0, [x8, x9, lsl #3]
	bl	_strlen
	str	x0, [sp, #16]
	ldur	x8, [x29, #-8]
	ldur	w9, [x29, #-32]
	add	w9, w9, #1
	ldr	x0, [x8, w9, sxtw #3]
	bl	_strlen
	str	x0, [sp, #8]
	ldr	x8, [sp, #16]
	ldr	x9, [sp, #8]
	subs	x8, x8, x9
	cset	w8, hi
	tbnz	w8, #0, LBB0_13
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_9 Depth=2
	ldr	x8, [sp, #16]
	ldr	x9, [sp, #8]
	subs	x8, x8, x9
	cset	w8, ne
	tbnz	w8, #0, LBB0_14
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_9 Depth=2
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-32]
	ldr	x0, [x8, x9, lsl #3]
	ldur	x8, [x29, #-8]
	ldur	w9, [x29, #-32]
	add	w9, w9, #1
	ldr	x1, [x8, w9, sxtw #3]
	bl	_strcmp
	subs	w8, w0, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_14
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_9 Depth=2
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-32]
	ldr	x8, [x8, x9, lsl #3]
	str	x8, [sp, #24]
	ldur	x8, [x29, #-8]
	ldur	w9, [x29, #-32]
	add	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldur	x9, [x29, #-8]
	ldursw	x10, [x29, #-32]
	str	x8, [x9, x10, lsl #3]
	ldr	x8, [sp, #24]
	ldur	x9, [x29, #-8]
	ldur	w10, [x29, #-32]
	add	w10, w10, #1
	str	x8, [x9, w10, sxtw #3]
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_9 Depth=2
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_9 Depth=2
	ldur	w8, [x29, #-32]
	add	w8, w8, #1
	stur	w8, [x29, #-32]
	b	LBB0_9
LBB0_16:                                ;   in Loop: Header=BB0_7 Depth=1
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_7 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_7
LBB0_18:
	ldur	x8, [x29, #-24]
	ldrsw	x9, [x8]
	mov	x8, #8
	mul	x0, x8, x9
	bl	_malloc
	str	x0, [sp]
	stur	wzr, [x29, #-28]
	b	LBB0_19
LBB0_19:                                ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-28]
	ldur	x9, [x29, #-24]
	ldr	w9, [x9]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_22
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_19 Depth=1
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-28]
	ldr	x8, [x8, x9, lsl #3]
	ldr	x9, [sp]
	ldursw	x10, [x29, #-28]
	str	x8, [x9, x11, lsl #3]
	b	LBB0_21
LBB0_21:                                ;   in Loop: Header=BB0_19 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_19
LBB0_22:
	ldr	x0, [sp]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

