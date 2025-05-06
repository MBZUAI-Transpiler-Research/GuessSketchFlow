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
	str	x3, [sp, #32]
	ldur	w8, [x29, #-12]
	ldr	x9, [sp, #32]
	str	w8, [x9]
	ldur	w8, [x29, #-12]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	ldur	x8, [x29, #-24]
                                        ; kill: def $x9 killed $xzr
	str	xzr, [x8]
	b	LBB0_23
LBB0_2:
	ldursw	x8, [x29, #-12]
	lsl	x0, x8, #2
	bl	_malloc
	ldur	x8, [x29, #-24]
	str	x0, [x8]
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	subs	x8, x8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:
	mov	w0, #1
	bl	_exit
LBB0_4:
	str	wzr, [sp, #28]
	b	LBB0_5
LBB0_5:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #28]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #28]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #28]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldr	w8, [sp, #28]
	add	w8, w8, #1
	str	w8, [sp, #28]
	b	LBB0_5
LBB0_8:
	ldur	x8, [x29, #-8]
	ldr	w8, [x8]
	ldur	x9, [x29, #-8]
	ldur	w10, [x29, #-12]
	subs	w10, w10, #1
	ldr	w9, [x9, w10, sxtw #2]
	add	w8, w8, w9
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #1
	cset	w8, eq
	and	w8, w8, #0x1
	str	w8, [sp, #24]
	str	wzr, [sp, #20]
	b	LBB0_9
LBB0_9:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_11 Depth 2
	ldr	w8, [sp, #22]
	ldur	w9, [x29, #-12]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_23
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_9 Depth=1
	ldr	w8, [sp, #22]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_11
LBB0_11:                                ;   Parent Loop BB0_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_21
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=2
	ldr	w8, [sp, #24]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_16
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #16]
	ldr	w9, [x9, x11, lsl #2]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_15
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #12]
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #22]
	str	w8, [x9, x10, lsl #2]
	ldr	w8, [sp, #12]
	ldur	x9, [x29, #-24]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #16]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_11 Depth=2
	b	LBB0_19
LBB0_16:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #16]
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_18
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #8]
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #22]
	str	w8, [x9, x10, lsl #2]
	ldr	w8, [sp, #8]
	ldur	x9, [x29, #-24]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #16]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_18
LBB0_18:                                ;   in Loop: Header=BB0_11 Depth=2
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_11 Depth=2
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_11 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_11
LBB0_21:                                ;   in Loop: Header=BB0_9 Depth=1
	b	LBB0_22
LBB0_22:                                ;   in Loop: Header=BB0_9 Depth=1
	ldr	w8, [sp, #22]
	add	w8, w8, #1
	str	w8, [sp, #22]
	b	LBB0_9
LBB0_23:
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

