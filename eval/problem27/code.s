	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
	stp	x29, x30, [sp, #112]            ; 16-byte Folded Spill
	add	x29, sp, #112
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
	ldursw	x0, [x29, #-12]
	mov	x1, #4
	str	x1, [sp]                        ; 8-byte Folded Spill
	bl	_calloc
	ldr	x1, [sp]                        ; 8-byte Folded Reload
	stur	x0, [x29, #-40]
	ldursw	x0, [x29, #-12]
	bl	_calloc
	stur	x0, [x29, #-48]
	stur	wzr, [x29, #-52]
	str	wzr, [sp, #56]
	str	wzr, [sp, #52]
	str	wzr, [sp, #48]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
                                        ;     Child Loop BB0_11 Depth 2
	ldr	w8, [sp, #48]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_21
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #48]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #44]
	str	wzr, [sp, #40]
	str	wzr, [sp, #36]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #36]
	ldr	w9, [sp, #56]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldur	x8, [x29, #-48]
	ldrsw	x9, [sp, #36]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #44]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #44]
	b	LBB0_8
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #36]
	add	w8, w8, #1
	str	w8, [sp, #36]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #44]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_20
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #32]
	str	wzr, [sp, #28]
	b	LBB0_11
LBB0_11:                                ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #28]
	ldur	w9, [x29, #-52]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_16
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-40]
	ldrsw	x9, [sp, #28]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #44]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_14
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #32]
	b	LBB0_16
LBB0_4:                                 ;   in Loop: Header=BB0_11 Depth=2
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_11 Depth=2
	ldr	w8, [sp, #28]
	add	w8, w8, #1
	str	w8, [sp, #28]
	b	LBB0_11
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #32]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #44]
	ldur	x9, [x29, #-48]
	ldrsw	x10, [sp, #56]
	mov	x11, x11
	add	w11, w11, #1
	str	w11, [sp, #56]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_9
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #44]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-52]
	mov	x11, x10
	add	w11, w11, #1
	stur	w11, [x29, #-52]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #48]
	add	w8, w8, #1
	str	w8, [sp, #48]
	b	LBB0_1
LBB0_11:
	str	wzr, [sp, #24]
	b	LBB0_12
LBB0_12:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_24 Depth 2
	ldr	w8, [sp, #24]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_33
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_22 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #20]
	str	wzr, [sp, #16]
	str	wzr, [sp, #12]
	b	LBB0_24
LBB0_24:                                ;   Parent Loop BB0_22 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #2]
	ldr	w9, [sp, #56]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_29
	b	LBB0_25
LBB0_25:                                ;   in Loop: Header=BB0_24 Depth=2
	ldur	x8, [x29, #-48]
	ldrsw	x9, [sp, #2]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #20]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_27
	b	LBB0_26
LBB0_26:                                ;   in Loop: Header=BB0_22 Depth=1
	mov	w8, #1
	str	w8, [sp, #16]
	b	LBB0_29
LBB0_27:                                ;   in Loop: Header=BB0_24 Depth=2
	b	LBB0_28
LBB0_28:                                ;   in Loop: Header=BB0_24 Depth=2
	ldr	w8, [sp, #2]
	add	w8, w8, #1
	str	w8, [sp, #2]
	b	LBB0_24
LBB0_29:                                ;   in Loop: Header=BB0_22 Depth=1
	ldr	w8, [sp, #16]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_31
	b	LBB0_30
LBB0_30:                                ;   in Loop: Header=BB0_22 Depth=1
	ldr	w8, [sp, #20]
	ldur	x9, [x29, #-32]
	ldrsw	x10, [sp, #52]
	mov	x11, x11
	add	w11, w11, #1
	str	w11, [sp, #52]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_31
LBB0_31:                                ;   in Loop: Header=BB0_22 Depth=1
	b	LBB0_32
LBB0_32:                                ;   in Loop: Header=BB0_22 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_22
LBB0_3:
	ldr	w8, [sp, #52]
	ldur	x9, [x29, #-24]
	str	w8, [x9]
	ldur	x0, [x29, #-40]
	bl	_free
	ldur	x0, [x29, #-48]
	bl	_free
	ldur	x0, [x29, #-2]
	ldp	x29, x30, [sp, #112]            ; 16-byte Folded Reload
	add	sp, sp, #128
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

