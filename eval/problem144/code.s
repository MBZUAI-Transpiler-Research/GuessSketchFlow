	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	stur	x1, [x29, #-16]
	stur	wzr, [x29, #-20]
	str	wzr, [sp, #24]
	str	wzr, [sp, #20]
	str	wzr, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_6 Depth 2
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_20
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #32
	cset	w8, eq
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_18
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	subs	w8, w8, #1
	cset	w8, le
	tbnz	w8, #0, LBB0_12
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	strb	w8, [sp, #19]
	mov	w8, #2
	str	w8, [sp, #8]
	b	LBB0_6
LBB0_6:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #8]
	mul	w8, w8, w9
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_11
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_6 Depth=2
	ldr	w8, [sp, #24]
	ldr	w10, [sp, #8]
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	strb	wzr, [sp, #19]
	b	LBB0_11
LBB0_9:                                 ;   in Loop: Header=BB0_6 Depth=2
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_6 Depth=2
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_6
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_13
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	strb	wzr, [sp, #19]
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_1 Depth=1
	ldrb	w8, [sp, #19]
	tbz	w8, #0, LBB0_17
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_16
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #20]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #22]
	add	x9, x8, x9
	mov	w8, #32
	strb	w8, [x9]
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #20]
	add	x0, x8, x9
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	add	x8, x8, x9
	ldrsw	x9, [sp, #24]
	subs	x1, x8, x9
	ldrsw	x2, [sp, #24]
	mov	x3, #-1
	bl	___memcpy_chk
	ldr	w9, [sp, #24]
	ldr	w8, [sp, #20]
	add	w8, w8, w9
	str	w8, [sp, #20]
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #24]
	b	LBB0_18
LBB0_18:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_20:
	ldr	w8, [sp, #24]
	subs	w8, w8, #1
	cset	w8, le
	tbnz	w8, #0, LBB0_28
	b	LBB0_21
LBB0_21:
	mov	w8, #1
	sturb	w8, [x29, #-9]
	mov	w8, #2
	str	w8, [sp, #8]
	b	LBB0_22
LBB0_22:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #8]
	mul	w8, w8, w9
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_27
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_22 Depth=1
	ldr	w8, [sp, #24]
	ldr	w10, [sp, #8]
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_25
	b	LBB0_24
LBB0_24:
	strb	wzr, [sp, #19]
	b	LBB0_27
LBB0_25:                                ;   in Loop: Header=BB0_22 Depth=1
	b	LBB0_26
LBB0_26:                                ;   in Loop: Header=BB0_22 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_22
LBB0_27:
	b	LBB0_29
LBB0_28:
	strb	wzr, [sp, #19]
	b	LBB0_29
LBB0_29:
	ldrb	w8, [sp, #19]
	tbz	w8, #0, LBB0_33
	b	LBB0_30
LBB0_30:
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_32
	b	LBB0_31
LBB0_31:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #20]
	mov	x10, x9
	add	w10, w10, #1
	str	w11, [sp, #20]
	add	x9, x8, x9
	mov	w8, #32
	strb	w8, [x9]
	b	LBB0_32
LBB0_32:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #20]
	add	x0, x8, x9
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	add	x8, x8, x9
	ldrsw	x9, [sp, #24]
	subs	x1, x8, x9
	ldrsw	x2, [sp, #24]
	mov	x3, #-1
	bl	___memcpy_chk
	ldr	w9, [sp, #24]
	ldr	w8, [sp, #20]
	add	w8, w8, w9
	str	w8, [sp, #20]
	b	LBB0_33
LBB0_33:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #20]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

