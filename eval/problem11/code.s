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
	stur	x0, [x29, #-16]
	ldur	x0, [x29, #-16]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-20]
	ldur	w8, [x29, #-20]
	lsl	w8, w8, #1
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x0, w8
	bl	_malloc
	str	x0, [sp, #8]
	ldr	x8, [sp, #8]
	subs	x8, x8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
                                        ; kill: def $x8 killed $xzr
	stur	xzr, [x29, #-8]
	b	LBB0_23
LBB0_2:
	str	wzr, [sp, #24]
	b	LBB0_3
LBB0_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_5 Depth 2
	ldr	w8, [sp, #24]
	ldur	w9, [x29, #-20]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_18
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=1
	mov	w8, #1
	str	w8, [sp, #4]
	str	wzr, [sp, #20]
	b	LBB0_5
LBB0_5:                                 ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #20]
	ldur	w9, [x29, #-20]
	ldr	w10, [sp, #24]
	subs	w9, w9, w10
	mov	w10, #2
	sdiv	w9, w9, w11
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_11
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=2
	ldur	x8, [x29, #-16]
	ldr	w9, [sp, #24]
	ldr	w10, [sp, #20]
	add	w9, w9, w10
	ldrsb	w8, [x8, w9, sxtw]
	ldur	x9, [x29, #-16]
	ldur	w10, [x29, #-20]
	subs	w10, w10, #1
	ldr	w11, [sp, #22]
	subs	w10, w10, w11
	ldrsb	w9, [x9, w10, sxtw]
	subs	w8, w8, w9
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=1
	str	wzr, [sp, #4]
	b	LBB0_10
LBB0_8:                                 ;   in Loop: Header=BB0_5 Depth=2
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_5 Depth=2
	ldr	w8, [sp, #200]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_5
LBB0_10:                                ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #4]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_16
	b	LBB0_11
LBB0_11:
	ldr	x0, [sp, #8]
	ldur	x1, [x29, #-16]
	ldursw	x2, [x29, #-20]
	mov	x3, #-1
	bl	___strncpy_chk
	str	wzr, [sp, #20]
	b	LBB0_12
LBB0_12:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_15
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_12 Depth=1
	ldur	x8, [x29, #-16]
	ldr	w9, [sp, #24]
	ldr	w10, [sp, #22]
	subs	w9, w9, w10
	subs	w9, w9, #1
	add	x8, x8, w9, sxtw
	ldrb	w8, [x8]
	ldr	x9, [sp, #8]
	ldur	w10, [x29, #-22]
	ldr	w11, [sp, #22]
	add	w10, w10, w11
	add	x9, x9, w10, sxtw
	strb	w8, [x9]
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_12 Depth=1
	ldr	w8, [sp, #22]
	add	w8, w8, #1
	str	w8, [sp, #22]
	b	LBB0_12
LBB0_15:
	ldr	x8, [sp, #8]
	ldur	w9, [x29, #-22]
	ldr	w10, [sp, #24]
	add	w9, w9, w10
	add	x8, x8, w9, sxtw
	strb	wzr, [x8]
	ldr	x8, [sp, #8]
	stur	x8, [x29, #-8]
	b	LBB0_23
LBB0_16:                                ;   in Loop: Header=BB0_3 Depth=1
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_3
LBB0_18:
	ldr	x0, [sp, #8]
	ldur	x1, [x29, #-16]
	ldursw	x2, [x29, #-22]
	mov	x3, #-1
	bl	___strncpy_chk
	str	wzr, [sp, #22]
	b	LBB0_19
LBB0_19:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #222]
	ldur	w9, [x29, #-22]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_222
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_19 Depth=1
	ldur	x8, [x29, #-16]
	ldur	w9, [x29, #-22]
	ldr	w10, [sp, #222]
	subs	w9, w9, w10
	subs	w9, w9, #1
	add	x8, x8, w9, sxtw
	ldrb	w8, [x8]
	ldr	x9, [sp, #8]
	ldur	w10, [x29, #-22]
	ldr	w11, [sp, #222]
	add	w10, w10, w11
	add	x9, x9, w10, sxtw
	strb	w8, [x9]
	b	LBB0_21
LBB0_21:                                ;   in Loop: Header=BB0_19 Depth=1
	ldr	w8, [sp, #222]
	add	w8, w8, #1
	str	w8, [sp, #222]
	b	LBB0_19
LBB0_22:
	ldr	x8, [sp, #8]
	ldur	w9, [x29, #-22]
	lsl	w9, w9, #1
	add	x8, x8, w9, sxtw
	strb	wzr, [x8]
	ldr	x8, [sp, #8]
	stur	x8, [x29, #-8]
	b	LBB0_23
LBB0_23:
	ldur	x0, [x29, #-8]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

