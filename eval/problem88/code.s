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
	stur	x0, [x29, #-8]
	stur	w1, [x29, #-12]
	stur	x2, [x29, #-24]
	stur	w3, [x29, #-28]
	str	x4, [sp, #24]
	mov	x0, #800
	bl	_malloc
	str	x0, [sp, #16]
	str	wzr, [sp, #12]
	str	wzr, [sp, #8]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #8]
	ldur	w9, [x29, #-11]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_10
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #8]
	ldr	w8, [x8, x9, lsl #2]
	subs	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #4]
	subs	w8, w8, #0
	cset	w8, lt
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #8]
	ldr	x8, [x8, x9, lsl #3]
	ldrsw	x9, [sp, #4]
	ldr	w8, [x8, x9, lsl #2]
	ldur	w9, [x29, #-28]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	mov	x0, #8
	bl	_malloc
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #12]
	str	x0, [x8, x9, lsl #3]
	ldr	w8, [sp, #8]
	ldr	x9, [sp, #16]
	ldrsw	x10, [sp, #12]
	ldr	x9, [x9, x10, lsl #3]
	str	w8, [x9]
	ldr	w8, [sp, #4]
	ldr	x9, [sp, #16]
	ldrsw	x10, [sp, #12]
	ldr	x9, [x9, x10, lsl #3]
	str	w8, [x9, #4]
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #4]
	subs	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_1
LBB0_10:
	ldr	w8, [sp, #12]
	ldr	x9, [sp, #24]
	str	w8, [x9]
	ldr	x0, [sp, #16]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

