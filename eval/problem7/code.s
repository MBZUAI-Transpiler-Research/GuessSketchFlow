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
                                        ; kill: def $x8 killed $xzr
	str	xzr, [sp, #24]
	str	wzr, [sp, #20]
	str	wzr, [sp, #16]
	str	wzr, [sp, #12]
	str	wzr, [sp, #8]
	str	wzr, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_13
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	add	x8, x8, x9
	ldrb	w8, [x8]
	strb	w8, [sp, #7]
	ldrsb	w8, [sp, #7]
	subs	w8, w8, #40
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #16]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	str	w8, [sp, #16]
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_11
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrsb	w8, [sp, #7]
	subs	w8, w8, #41
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	subs	w8, w8, #1
	str	w8, [sp, #20]
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x0, [sp, #24]
	ldr	w8, [sp, #8]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	lsl	x1, x8, #2
	bl	_realloc
	str	x0, [sp, #24]
	ldr	w8, [sp, #16]
	ldr	x9, [sp, #24]
	ldrsw	x10, [sp, #8]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp, #8]
	str	w8, [x9, x10, lsl #2]
	str	wzr, [sp, #16]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_13:
	ldr	w8, [sp, #8]
	ldur	x9, [x29, #-16]
	str	w8, [x9]
	ldr	x0, [sp, #24]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

