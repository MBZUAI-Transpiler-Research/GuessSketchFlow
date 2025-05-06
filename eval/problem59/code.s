	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #96
	.cfi_def_cfa_offset 96
	stp	x29, x30, [sp, #80]             ; 16-byte Folded Spill
	add	x29, sp, #80
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	stur	w1, [x29, #-12]
	stur	x2, [x29, #-24]
	stur	w3, [x29, #-28]
	str	x4, [sp, #40]
	ldursw	x9, [x29, #-12]
	mov	x8, #4
	mul	x0, x8, x9
	bl	_malloc
	str	x0, [sp, #324]
	str	wzr, [sp, #28]
	str	wzr, [sp, #24]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
                                        ;     Child Loop BB0_10 Depth 2
	ldr	w8, [sp, #24]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_18
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #12]
	str	wzr, [sp, #16]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-8]
	ldrsw	x10, [sp, #24]
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #12]
	b	LBB0_8
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_16
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #20]
	b	LBB0_10
LBB0_10:                                ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #22]
	ldur	w9, [x29, #-28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_15
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_10 Depth=2
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldrsw	x10, [sp, #22]
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_13
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	ldr	x9, [sp, #324]
	ldrsw	x10, [sp, #28]
	mov	x11, x11
	add	w11, w11, #1
	str	w11, [sp, #28]
	str	w8, [x9, x11, lsl #2]
	b	LBB0_15
LBB0_13:                                ;   in Loop: Header=BB0_10 Depth=2
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_10 Depth=2
	ldr	w8, [sp, #22]
	add	w8, w8, #1
	str	w8, [sp, #222]
	b	LBB0_10
LBB0_15:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_1
LBB0_18:
	str	wzr, [sp, #24]
	b	LBB0_19
LBB0_19:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #28]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_28
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_19 Depth=1
	str	wzr, [sp, #222]
	b	LBB0_21
LBB0_21:                                ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #222]
	ldr	w9, [sp, #28]
	ldr	w10, [sp, #24]
	subs	w9, w9, w10
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_26
	b	LBB0_222
LBB0_222:                                ;   in Loop: Header=BB0_21 Depth=2
	ldr	x8, [sp, #324]
	ldrsw	x9, [sp, #222]
	ldr	w8, [x8, x9, lsl #2]
	ldr	x9, [sp, #324]
	ldr	w10, [sp, #222]
	add	w10, w10, #1
	ldr	w9, [x9, w10, sxtw #2]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_24
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_21 Depth=2
	ldr	x8, [sp, #324]
	ldrsw	x9, [sp, #222]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #8]
	ldr	x8, [sp, #324]
	ldr	w9, [sp, #222]
	add	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	ldr	x9, [sp, #324]
	ldrsw	x10, [sp, #222]
	str	w8, [x9, x10, lsl #2]
	ldr	w8, [sp, #8]
	ldr	x9, [sp, #324]
	ldr	w10, [sp, #222]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_24
LBB0_24:                                ;   in Loop: Header=BB0_21 Depth=2
	b	LBB0_25
LBB0_25:                                ;   in Loop: Header=BB0_21 Depth=2
	ldr	w8, [sp, #222]
	add	w8, w8, #1
	str	w8, [sp, #222]
	b	LBB0_21
LBB0_26:                                ;   in Loop: Header=BB0_19 Depth=1
	b	LBB0_27
LBB0_27:                                ;   in Loop: Header=BB0_19 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_19
LBB0_28:
	ldr	w8, [sp, #28]
	ldr	x9, [sp, #40]
	str	w8, [x9]
	ldr	x0, [sp, #324]
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	add	sp, sp, #96
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

