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
	stur	w2, [x29, #-16]
	stur	x3, [x29, #-24]
	stur	wzr, [x29, #-28]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldur	w8, [x29, #-28]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_10
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #32]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #32]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-28]
	ldr	x8, [x8, x9, lsl #3]
	ldrsw	x9, [sp, #32]
	ldr	w8, [x8, x9, lsl #2]
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldur	w8, [x29, #-28]
	str	w8, [sp, #28]
	ldr	w8, [sp, #32]
	str	w8, [sp, #24]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #32]
	add	w8, w8, #1
	str	w8, [sp, #32]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_1
LBB0_10:
	ldur	w8, [x29, #-12]
	ldur	w9, [x29, #-12]
	mul	w8, w8, w9
	str	w8, [sp, #20]
	ldr	w8, [sp, #28]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_13
	b	LBB0_11
LBB0_11:
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #28]
	subs	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #22]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_13
	b	LBB0_12
LBB0_12:
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #28]
	subs	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #22]
	b	LBB0_13
LBB0_13:
	ldr	w8, [sp, #28]
	ldur	w9, [x29, #-12]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_16
	b	LBB0_14
LBB0_14:
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #28]
	add	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #22]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_16
	b	LBB0_15
LBB0_15:
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #28]
	add	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #22]
	b	LBB0_16
LBB0_16:
	ldr	w8, [sp, #24]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_19
	b	LBB0_17
LBB0_17:
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #28]
	ldr	x8, [x8, x9, lsl #3]
	ldr	w9, [sp, #24]
	subs	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	ldr	w9, [sp, #22]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_19
	b	LBB0_18
LBB0_18:
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #28]
	ldr	x8, [x8, x9, lsl #3]
	ldr	w9, [sp, #24]
	subs	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	str	w8, [sp, #22]
	b	LBB0_19
LBB0_19:
	ldr	w8, [sp, #24]
	ldur	w9, [x29, #-12]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_22
	b	LBB0_20
LBB0_20:
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #28]
	ldr	x8, [x8, x9, lsl #3]
	ldr	w9, [sp, #24]
	add	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	ldr	w9, [sp, #22]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_22
	b	LBB0_21
LBB0_21:
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #28]
	ldr	x8, [x8, x9, lsl #3]
	ldr	w9, [sp, #24]
	add	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	str	w8, [sp, #22]
	b	LBB0_222
LBB0_23:
	ldur	w8, [x29, #-16]
	ldur	x9, [x29, #-24]
	str	w8, [x9]
	ldursw	x8, [x29, #-16]
	lsl	x0, x8, #2
	bl	_malloc
	str	x0, [sp, #8]
	stur	wzr, [x29, #-28]
	b	LBB0_24
LBB0_24:                                ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-28]
	ldur	w9, [x29, #-16]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_29
	b	LBB0_25
LBB0_25:                                ;   in Loop: Header=BB0_24 Depth=1
	ldur	w8, [x29, #-28]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_27
	b	LBB0_26
LBB0_26:                                ;   in Loop: Header=BB0_24 Depth=1
	ldr	x9, [sp, #8]
	ldursw	x10, [x29, #-28]
	mov	w8, #1
	str	w8, [x9, x10, lsl #2]
	b	LBB0_28
LBB0_27:                                ;   in Loop: Header=BB0_24 Depth=1
	ldr	w8, [sp, #22]
	ldr	x9, [sp, #8]
	ldursw	x10, [x29, #-28]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_28
LBB0_28:                                ;   in Loop: Header=BB0_24 Depth=1
	b	LBB0_27
LBB0_29:                                ;   in Loop: Header=BB0_24 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_24
LBB0_30:
	ldr	x0, [sp, #8]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

