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
	stur	x0, [x29, #-16]
	stur	x1, [x29, #-24]
	ldur	x0, [x29, #-16]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-28]
	ldur	x0, [x29, #-24]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-32]
	str	wzr, [sp, #28]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #28]
	ldur	w9, [x29, #-28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_12
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	strb	wzr, [sp, #27]
	str	wzr, [sp, #20]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #20]
	ldur	w9, [x29, #-32]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #28]
	ldrsb	w8, [x8, x9]
	ldur	x9, [x29, #-24]
	ldrsw	x10, [sp, #20]
	ldrsb	w9, [x9, x10]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	strb	w8, [sp, #27]
	b	LBB0_8
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrb	w8, [sp, #27]
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	mov	w8, #0
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	sturb	w8, [x29, #-1]
	b	LBB0_25
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #28]
	add	w8, w8, #1
	str	w8, [sp, #28]
	b	LBB0_1
LBB0_12:
	str	wzr, [sp, #16]
	b	LBB0_13
LBB0_13:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_15 Depth 2
	ldr	w8, [sp, #16]
	ldur	w9, [x29, #-32]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_24
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_13 Depth=1
	strb	wzr, [sp, #15]
	str	wzr, [sp, #8]
	b	LBB0_15
LBB0_15:                                ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #8]
	ldur	w9, [x29, #-28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_20
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_15 Depth=2
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #16]
	ldrsb	w8, [x8, x9]
	ldur	x9, [x29, #-16]
	ldrsw	x10, [sp, #8]
	ldrsb	w9, [x9, x10]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_18
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_13 Depth=1
	mov	w8, #1
	strb	w8, [sp, #15]
	b	LBB0_22
LBB0_18:                                ;   in Loop: Header=BB0_15 Depth=2
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_15 Depth=2
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_15
LBB0_20:                                ;   in Loop: Header=BB0_13 Depth=1
	ldrb	w8, [sp, #15]
	tbnz	w8, #0, LBB0_24
	b	LBB0_23
LBB0_23:
	mov	w8, #0
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	sturb	w8, [x29, #-1]
	b	LBB0_25
LBB0_24:                                ;   in Loop: Header=BB0_13 Depth=1
	b	LBB0_23
LBB0_25:                                ;   in Loop: Header=BB0_13 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_13
LBB0_26:
	mov	w8, #1
	and	w8, w8, #0x1
	and	w8, w8, #0x1
	sturb	w8, [x29, #-1]
	b	LBB0_27
LBB0_27:
	ldurb	w8, [x29, #-1]
	and	w0, w8, #0x1
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

