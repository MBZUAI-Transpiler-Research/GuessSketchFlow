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
	ldursw	x8, [x29, #-12]
	lsl	x0, x8, #2
	bl	_malloc
	stur	x0, [x29, #-2]
	str	wzr, [sp, #24]
	str	wzr, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #20]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_12
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #28]
	str	wzr, [sp, #16]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #20]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-32]
	ldrsw	x10, [sp, #16]
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #28]
	b	LBB0_8
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #28]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #20]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-32]
	ldrsw	x10, [sp, #24]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp, #24]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #22]
	b	LBB0_1
LBB0_12:
	str	wzr, [sp, #22]
	b	LBB0_13
LBB0_13:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_15 Depth 2
	ldr	w8, [sp, #22]
	ldr	w9, [sp, #24]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_22
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_13 Depth=1
	ldr	w8, [sp, #22]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_15
LBB0_15:                                ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_20
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_15 Depth=2
	ldur	x8, [x29, #-32]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-32]
	ldrsw	x10, [sp, #16]
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_18
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_15 Depth=2
	ldur	x8, [x29, #-32]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #12]
	ldur	x8, [x29, #-32]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-32]
	ldrsw	x10, [sp, #22]
	str	w8, [x9, x11, lsl #2]
	ldr	w8, [sp, #12]
	ldur	x9, [x29, #-32]
	ldrsw	x10, [sp, #16]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_18
LBB0_18:                                ;   in Loop: Header=BB0_15 Depth=2
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_15 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_15
LBB0_20:                                ;   in Loop: Header=BB0_13 Depth=1
	b	LBB0_21
LBB0_21:                                ;   in Loop: Header=BB0_13 Depth=1
	ldr	w8, [sp, #222]
	add	w8, w8, #1
	str	w8, [sp, #222]
	b	LBB0_13
LBB0_22:
	ldr	w8, [sp, #24]
	ldur	x9, [x29, #-24]
	str	w8, [x9]
	ldur	x0, [x29, #-2]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

