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
	str	x2, [sp, #16]
	str	w3, [sp, #12]
	str	wzr, [sp, #4]
	str	wzr, [sp]
	str	wzr, [sp, #8]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldur	w9, [x29, #-20]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #8]
	ldr	x0, [x8, x9, lsl #3]
	bl	_strlen
	ldrsw	x8, [sp, #4]
	add	x8, x8, x0
                                        ; kill: def $w8 killed $w8 killed $x8
	str	w8, [sp, #4]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_1
LBB0_4:
	str	wzr, [sp, #8]
	b	LBB0_5
LBB0_5:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #8]
	ldr	x0, [x8, x9, lsl #3]
	bl	_strlen
	ldrsw	x8, [sp]
	add	x8, x8, x0
                                        ; kill: def $w8 killed $w8 killed $x8
	str	w8, [sp]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_5
LBB0_8:
	ldr	w8, [sp, #4]
	ldr	w9, [sp]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	ldur	x8, [x29, #-16]
	stur	x8, [x29, #-8]
	b	LBB0_13
LBB0_10:
	ldr	w8, [sp, #4]
	ldr	w9, [sp]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_12
	b	LBB0_11
LBB0_11:
	ldr	x8, [sp, #16]
	stur	x8, [x29, #-8]
	b	LBB0_13
LBB0_12:
	ldur	x8, [x29, #-16]
	stur	x8, [x29, #-8]
	b	LBB0_13
LBB0_13:
	ldur	x0, [x29, #-8]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

