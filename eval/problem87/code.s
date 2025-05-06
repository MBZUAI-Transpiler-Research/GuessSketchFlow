	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #144
	.cfi_def_cfa_offset 144
	stp	x29, x30, [sp, #128]            ; 16-byte Folded Spill
	add	x29, sp, #128
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-8]
	str	x0, [sp, #56]
	ldr	x0, [sp, #56]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #52]
	ldr	w8, [sp, #52]
	add	w9, w8, #2
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x0, w8
	bl	_malloc
	str	x0, [sp, #40]
	str	wzr, [sp, #36]
	str	wzr, [sp, #32]
	str	wzr, [sp, #28]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_5 Depth 2
                                        ;       Child Loop BB0_7 Depth 3
                                        ;     Child Loop BB0_17 Depth 2
	ldr	w8, [sp, #28]
	ldr	w9, [sp, #52]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_24
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #56]
	ldrsw	x9, [sp, #28]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #32
	cset	w8, eq
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #56]
	ldrsw	x9, [sp, #28]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_21
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #24]
	b	LBB0_5
LBB0_5:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_7 Depth 3
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #32]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_14
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=2
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_7
LBB0_7:                                 ;   Parent Loop BB0_1 Depth=1
                                        ;     Parent Loop BB0_5 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #32]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_12
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_7 Depth=3
	ldrsw	x8, [sp, #24]
	sub	x9, x29, #58
	ldrsb	w8, [x9, x8]
	ldrsw	x10, [sp, #22]
	ldrsb	w9, [x9, x10]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_7 Depth=3
	ldrsw	x10, [sp, #24]
	sub	x9, x29, #58
	mov	x8, x9
	add	x8, x8, x11
	ldrb	w8, [x8]
	strb	w8, [sp, #21]
	ldrsw	x10, [sp, #222]
	mov	x8, x9
	add	x8, x8, x11
	ldrb	w8, [x8]
	ldrsw	x11, [sp, #24]
	mov	x10, x9
	add	x10, x10, x11
	strb	w8, [x10]
	ldrb	w8, [sp, #21]
	ldrsw	x10, [sp, #222]
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_7 Depth=3
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_7 Depth=3
	ldr	w8, [sp, #22]
	add	w8, w8, #1
	str	w8, [sp, #222]
	b	LBB0_7
LBB0_12:                                ;   in Loop: Header=BB0_5 Depth=2
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_5 Depth=2
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_5
LBB0_14:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #36]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_6
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #36]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #36]
	add	x9, x8, x9
	mov	w8, #33
	strb	w8, [x9]
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #16]
	b	LBB0_17
LBB0_17:                                ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #32]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_20
	b	LBB0_18
LBB0_18:                                ;   in Loop: Header=BB0_17 Depth=2
	ldrsw	x9, [sp, #16]
	sub	x8, x29, #58
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldr	x9, [sp, #40]
	ldrsw	x10, [sp, #36]
	mov	x11, x11
	add	w11, w11, #1
	str	w11, [sp, #36]
	add	x9, x9, x11
	strb	w8, [x9]
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_17 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_17
LBB0_20:                                ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #32]
	b	LBB0_22
LBB0_21:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #56]
	ldrsw	x9, [sp, #28]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #32]
	mov	x9, x11
	add	w9, w9, #1
	str	w9, [sp, #32]
	sub	x9, x29, #58
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_22
LBB0_22:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #28]
	add	w8, w8, #1
	str	w8, [sp, #28]
	b	LBB0_1
LBB0_24:
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #36]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldr	x8, [sp, #40]
	str	x8, [sp]                        ; 8-byte Folded Spill
	ldur	x9, [x29, #-8]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_26
	b	LBB0_25
LBB0_25:
	bl	___stack_chk_fail
LBB0_26:
	ldr	x0, [sp]                        ; 8-byte Folded Reload
	ldp	x29, x30, [sp, #128]            ; 16-byte Folded Reload
	add	sp, sp, #144
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

