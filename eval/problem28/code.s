	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	str	x1, [sp, #16]
	ldur	x0, [x29, #-8]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #12]
	str	wzr, [sp, #8]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_11
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #8]
	add	x8, x8, x9
	ldrb	w8, [x8]
	strb	w8, [sp, #7]
	ldrsb	w8, [sp, #7]
	subs	w8, w8, #97
	cset	w8, lt
	tbnz	w8, #0, LBB0_5
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrsb	w8, [sp, #7]
	subs	w8, w8, #122
	cset	w8, gt
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrsb	w8, [sp, #7]
	subs	w8, w8, #32
	strb	w8, [sp, #7]
	b	LBB0_9
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrsb	w8, [sp, #7]
	subs	w8, w8, #65
	cset	w8, lt
	tbnz	w8, #0, LBB0_8
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrsb	w8, [sp, #7]
	subs	w8, w8, #90
	cset	w8, gt
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrsb	w8, [sp, #7]
	add	w8, w8, #32
	strb	w8, [sp, #7]
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrb	w8, [sp, #7]
	ldr	x9, [sp, #16]
	ldrsw	x10, [sp, #8]
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_1
LBB0_11:
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #12]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

