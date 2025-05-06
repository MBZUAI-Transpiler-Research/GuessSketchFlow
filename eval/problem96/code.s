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
	stur	w1, [x29, #-20]
	ldur	w8, [x29, #-20]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	stur	wzr, [x29, #-4]
	b	LBB0_19
LBB0_2:
	str	wzr, [sp, #24]
	str	wzr, [sp, #20]
	str	wzr, [sp, #16]
	b	LBB0_3
LBB0_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_5 Depth 2
	ldr	w8, [sp, #16]
	ldur	w9, [x29, #-20]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_18
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #16]
	lsl	x9, x9, #4
	ldr	x8, [x8, x9]
	str	x8, [sp, #8]
	str	wzr, [sp, #4]
	b	LBB0_5
LBB0_5:                                 ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [sp, #8]
	ldrsw	x9, [sp, #4]
	add	x8, x8, x9
	ldrb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_16
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=2
	ldr	x8, [sp, #8]
	ldrsw	x9, [sp, #4]
	ldrb	w0, [x8, x9]
	bl	_isalpha
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:
	stur	wzr, [x29, #-4]
	b	LBB0_19
LBB0_8:                                 ;   in Loop: Header=BB0_5 Depth=2
	ldr	x8, [sp, #8]
	ldrsw	x9, [sp, #4]
	ldrb	w0, [x8, x9]
	bl	_isupper
	subs	w8, w0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_5 Depth=2
	mov	w8, #1
	str	w8, [sp, #20]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_5 Depth=2
	ldr	x8, [sp, #8]
	ldrsw	x9, [sp, #4]
	ldrb	w0, [x8, x9]
	bl	_islower
	subs	w8, w0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_12
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_5 Depth=2
	mov	w8, #1
	str	w8, [sp, #24]
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_5 Depth=2
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #24]
	add	w8, w8, w9
	subs	w8, w8, #2
	cset	w8, ne
	tbnz	w8, #0, LBB0_14
	b	LBB0_13
LBB0_13:
	stur	wzr, [x29, #-4]
	b	LBB0_19
LBB0_14:                                ;   in Loop: Header=BB0_5 Depth=2
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_5 Depth=2
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_5
LBB0_16:                                ;   in Loop: Header=BB0_3 Depth=1
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_3
LBB0_18:
	mov	w8, #1
	stur	w8, [x29, #-4]
	b	LBB0_19
LBB0_19:
	ldur	w0, [x29, #-4]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

