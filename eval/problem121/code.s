	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, #40]
	str	w1, [sp, #36]
	str	w2, [sp, #32]
	str	x3, [sp, #24]
	str	wzr, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #36]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_10
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #16]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #36]
	ldr	w10, [sp, #20]
	subs	w9, w9, w10
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	ldr	x9, [sp, #40]
	ldr	w10, [sp, #16]
	add	w10, w10, #1
	ldr	w9, [x9, w10, sxtw #2]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #12]
	ldr	x8, [sp, #40]
	ldr	w9, [sp, #16]
	add	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	ldr	x9, [sp, #40]
	ldrsw	x10, [sp, #16]
	str	w8, [x9, x10, lsl #2]
	ldr	w8, [sp, #12]
	ldr	x9, [sp, #40]
	ldr	w10, [sp, #16]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_1
LBB0_10:
	str	wzr, [sp, #8]
	b	LBB0_11
LBB0_11:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #32]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_14
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	x8, [sp, #40]
	ldr	w9, [sp, #36]
	ldr	w10, [sp, #32]
	subs	w9, w9, w10
	ldr	w10, [sp, #8]
	add	w9, w9, w10
	ldr	w8, [x8, w9, sxtw #2]
	ldr	x9, [sp, #24]
	ldrsw	x10, [sp, #8]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_11
LBB0_14:
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

