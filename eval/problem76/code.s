	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	w0, [sp, #8]
	ldr	w8, [sp, #8]
	subs	w8, w8, #2
	cset	w8, ge
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	str	wzr, [sp, #12]
	b	LBB0_12
LBB0_2:
	str	wzr, [sp, #4]
	mov	w8, #2
	str	w8, [sp]
	b	LBB0_3
LBB0_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_5 Depth 2
	ldr	w8, [sp]
	ldr	w9, [sp]
	mul	w8, w8, w9
	ldr	w9, [sp, #8]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_9
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=1
	b	LBB0_5
LBB0_5:                                 ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #8]
	ldr	w10, [sp]
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=2
	ldr	w8, [sp, #8]
	ldr	w9, [sp]
	sdiv	w8, w8, w9
	str	w8, [sp, #8]
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_5
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=1
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp]
	add	w8, w8, #1
	str	w8, [sp]
	b	LBB0_3
LBB0_9:
	ldr	w8, [sp, #8]
	subs	w8, w8, #1
	cset	w8, le
	tbnz	w8, #0, LBB0_11
	b	LBB0_10
LBB0_10:
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_11
LBB0_11:
	ldr	w8, [sp, #4]
	subs	w8, w8, #3
	cset	w8, eq
	and	w8, w8, #0x1
	str	w8, [sp, #12]
	b	LBB0_12
LBB0_12:
	ldr	w0, [sp, #12]
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

