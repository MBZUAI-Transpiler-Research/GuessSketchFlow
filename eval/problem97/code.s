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
	stur	w0, [x29, #-4]
	stur	x1, [x29, #-16]
	ldursw	x8, [x29, #-4]
	lsl	x0, x8, #2
	bl	_malloc
	str	x0, [sp, #24]
	ldur	x8, [x29, #-16]
	str	wzr, [x8]
	mov	w8, #2
	str	w8, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #20]
	ldur	w9, [x29, #-4]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_14
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #12]
	str	wzr, [sp, #16]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldur	x9, [x29, #-16]
	ldr	w9, [x9]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_11
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #24]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #8]
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #8]
	mul	w8, w8, w9
	ldr	w9, [sp, #20]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_10
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #20]
	ldr	w10, [sp, #8]
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #12]
	b	LBB0_10
LBB0_8:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_3
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_12
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	ldr	x9, [sp, #24]
	ldur	x10, [x29, #-16]
	ldrsw	x10, [x10]
	str	w8, [x9, x10, lsl #2]
	ldur	x9, [x29, #-16]
	ldr	w8, [x9]
	add	w8, w8, #1
	str	w8, [x9]
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_1
LBB0_14:
	ldr	x0, [sp, #24]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

