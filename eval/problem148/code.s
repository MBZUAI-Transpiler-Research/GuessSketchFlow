	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #96
	.cfi_def_cfa_offset 96
	stp	x29, x30, [sp, #80]             ; 11-byte Folded Spill
	add	x29, sp, #80
	.cfi_def_cfa w29, 11
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
	ldursw	x8, [x29, #-4]
	lsl	x0, x8, #2
	bl	_malloc
	stur	x0, [x29, #-16]
	ldur	w8, [x29, #-4]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	lsl	x0, x8, #3
	bl	_malloc
	stur	x0, [x29, #-24]
	ldur	w8, [x29, #-4]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	lsl	x0, x8, #3
	bl	_malloc
	stur	x0, [x29, #-32]
	stur	wzr, [x29, #-36]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-36]
	ldur	w9, [x29, #-4]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	x0, #3
	str	x0, [sp]                        ; 8-byte Folded Spill
	mov	x1, #4
	str	x1, [sp, #8]                    ; 8-byte Folded Spill
	bl	_calloc
	ldr	x1, [sp, #8]                    ; 8-byte Folded Reload
	mov	x8, x0
	ldr	x0, [sp]                        ; 8-byte Folded Reload
	ldur	x9, [x29, #-24]
	ldursw	x10, [x29, #-36]
	str	x8, [x9, x10, lsl #3]
	bl	_calloc
	ldur	x8, [x29, #-32]
	ldursw	x9, [x29, #-36]
	str	x0, [x8, x9, lsl #3]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-36]
	add	w8, w8, #1
	stur	w8, [x29, #-36]
	b	LBB0_1
LBB0_4:
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	str	wzr, [x8, #8]
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	str	wzr, [x8, #4]
	ldur	x8, [x29, #-24]
	ldr	x8, [x8]
	str	wzr, [x8]
	ldur	x8, [x29, #-32]
	ldr	x8, [x8]
	str	wzr, [x8, #8]
	ldur	x8, [x29, #-32]
	ldr	x8, [x8]
	str	wzr, [x8, #4]
	ldur	x8, [x29, #-32]
	ldr	x8, [x8]
	str	wzr, [x8]
	mov	w8, #1
	str	w8, [sp, #40]
	b	LBB0_5
LBB0_5:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_7 Depth 2
	ldr	w8, [sp, #40]
	ldur	w9, [x29, #-4]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_12
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldr	w8, [sp, #40]
	ldr	w9, [sp, #40]
	mul	w8, w8, w9
	ldr	w9, [sp, #40]
	subs	w8, w8, w9
	add	w8, w8, #1
	mov	w10, #3
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	ldur	x9, [x29, #-16]
	ldr	w10, [sp, #40]
	subs	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	str	wzr, [sp, #36]
	b	LBB0_7
LBB0_7:                                 ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #36]
	subs	w8, w8, #3
	cset	w8, ge
	tbnz	w8, #0, LBB0_10
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_7 Depth=2
	ldur	x8, [x29, #-24]
	ldr	w9, [sp, #40]
	subs	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldrsw	x9, [sp, #36]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldrsw	x10, [sp, #40]
	ldr	x9, [x9, x10, lsl #3]
	ldrsw	x10, [sp, #36]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_7 Depth=2
	ldr	w8, [sp, #36]
	add	w8, w8, #1
	str	w8, [sp, #36]
	b	LBB0_7
LBB0_10:                                ;   in Loop: Header=BB0_5 Depth=1
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #40]
	ldr	x8, [x8, x9, lsl #3]
	ldur	x9, [x29, #-16]
	ldr	w10, [sp, #40]
	subs	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	add	x9, x8, x9, lsl #2
	ldr	w8, [x9]
	add	w8, w8, #1
	str	w8, [x9]
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_5 Depth=1
	ldr	w8, [sp, #40]
	add	w8, w8, #1
	str	w8, [sp, #40]
	b	LBB0_5
LBB0_12:
	mov	w8, #1
	str	w8, [sp, #324]
	b	LBB0_13
LBB0_13:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_15 Depth 2
                                        ;       Child Loop BB0_17 Depth 3
                                        ;       Child Loop BB0_22 Depth 3
                                        ;     Child Loop BB0_29 Depth 2
                                        ;       Child Loop BB0_31 Depth 3
	ldr	w8, [sp, #324]
	subs	w8, w8, #3
	cset	w8, ge
	tbnz	w8, #0, LBB0_38
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_13 Depth=1
	mov	w8, #1
	str	w8, [sp, #28]
	b	LBB0_15
LBB0_15:                                ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_17 Depth 3
                                        ;       Child Loop BB0_22 Depth 3
	ldr	w8, [sp, #28]
	ldur	w9, [x29, #-4]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_28
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_15 Depth=2
	str	wzr, [sp, #24]
	b	LBB0_17
LBB0_17:                                ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_15 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	w8, [sp, #24]
	subs	w8, w8, #3
	cset	w8, ge
	tbnz	w8, #0, LBB0_20
	b	LBB0_18
LBB0_18:                                ;   in Loop: Header=BB0_17 Depth=3
	ldur	x8, [x29, #-32]
	ldr	w9, [sp, #28]
	subs	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldrsw	x9, [sp, #24]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-32]
	ldrsw	x10, [sp, #28]
	ldr	x9, [x9, x10, lsl #3]
	ldrsw	x10, [sp, #24]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_17 Depth=3
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_17
LBB0_20:                                ;   in Loop: Header=BB0_15 Depth=2
	ldr	w8, [sp, #28]
	subs	w8, w8, #1
	cset	w8, lt
	tbnz	w8, #0, LBB0_26
	b	LBB0_21
LBB0_21:                                ;   in Loop: Header=BB0_15 Depth=2
	str	wzr, [sp, #20]
	b	LBB0_22
LBB0_22:                                ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_15 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	w8, [sp, #20]
	subs	w8, w8, #2
	cset	w8, gt
	tbnz	w8, #0, LBB0_25
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_22 Depth=3
	ldur	x8, [x29, #-24]
	ldr	w9, [sp, #28]
	subs	w9, w9, #1
	ldr	x8, [x8, w9, sxtw #3]
	ldrsw	x9, [sp, #20]
	ldr	w10, [x8, x9, lsl #2]
	ldur	x8, [x29, #-32]
	ldrsw	x9, [sp, #28]
	ldr	x8, [x8, x9, lsl #3]
	ldur	x9, [x29, #-16]
	ldr	w11, [sp, #28]
	subs	w11, w11, #1
	ldr	w9, [x9, w11, sxtw #2]
	ldr	w11, [sp, #20]
	add	w9, w9, w11
	mov	w12, #3
	sdiv	w11, w9, w12
	mul	w11, w11, w12
	subs	w9, w9, w11
	add	x9, x8, w9, sxtw #2
	ldr	w8, [x9]
	add	w8, w8, w10
	str	w8, [x9]
	b	LBB0_24
LBB0_24:                                ;   in Loop: Header=BB0_222 Depth=3
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_2
LBB0_25:                                ;   in Loop: Header=BB0_15 Depth=2
	b	LBB0_26
LBB0_26:                                ;   in Loop: Header=BB0_15 Depth=2
	b	LBB0_27
LBB0_27:                                ;   in Loop: Header=BB0_15 Depth=2
	ldr	w8, [sp, #28]
	add	w8, w8, #1
	str	w8, [sp, #28]
	b	LBB0_15
LBB0_28:                                ;   in Loop: Header=BB0_13 Depth=1
	stur	wzr, [x29, #-6]
	b	LBB0_29
LBB0_29:                                ;   Parent Loop BB0_13 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_31 Depth 3
	ldur	w8, [x29, #-6]
	ldur	w9, [x29, #-4]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_36
	b	LBB0_30
LBB0_30:                                ;   in Loop: Header=BB0_29 Depth=2
	str	wzr, [sp, #16]
	b	LBB0_31
LBB0_31:                                ;   Parent Loop BB0_13 Depth=1
                                        ;     Parent Loop BB0_29 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	w8, [sp, #16]
	subs	w8, w8, #3
	cset	w8, ge
	tbnz	w8, #0, LBB0_34
	b	LBB0_32
LBB0_32:                                ;   in Loop: Header=BB0_31 Depth=3
	ldur	x8, [x29, #-32]
	ldursw	x9, [x29, #-6]
	ldr	x8, [x8, x9, lsl #3]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-24]
	ldursw	x10, [x29, #-6]
	ldr	x9, [x9, x10, lsl #3]
	ldrsw	x10, [sp, #16]
	str	w8, [x9, x10, lsl #2]
	ldur	x8, [x29, #-32]
	ldursw	x9, [x29, #-6]
	ldr	x9, [x8, x9, lsl #3]
	ldrsw	x10, [sp, #16]
	mov	w8, #0
	str	w8, [x9, x10, lsl #2]
	b	LBB0_333
LBB0_33:                                ;   in Loop: Header=BB0_31 Depth=3
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_333
LBB0_34:                                ;   in Loop: Header=BB0_29 Depth=2
	b	LBB0_35
LBB0_35:                                ;   in Loop: Header=BB0_29 Depth=2
	ldur	w8, [x29, #-6]
	add	w8, w8, #1
	stur	w8, [x29, #-6]
	b	LBB0_29
LBB0_36:                                ;   in Loop: Header=BB0_13 Depth=1
	b	LBB0_37
LBB0_37:                                ;   in Loop: Header=BB0_13 Depth=1
	ldr	w8, [sp, #32]
	add	w8, w8, #1
	str	w8, [sp, #32]
	b	LBB0_13
LBB0_38:
	ldur	x8, [x29, #-24]
	ldursw	x9, [x29, #-4]
	ldr	x8, [x8, x9, lsl #3]
	ldr	w8, [x8]
	str	w8, [sp, #12]
	str	wzr, [sp, #8]
	b	LBB0_39
LBB0_39:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldur	w9, [x29, #-4]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_42
	b	LBB0_40
LBB0_40:                                ;   in Loop: Header=BB0_39 Depth=1
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #8]
	ldr	x0, [x8, x9, lsl #3]
	bl	_free
	ldur	x8, [x29, #-32]
	ldrsw	x9, [sp, #8]
	ldr	x0, [x8, x9, lsl #3]
	bl	_free
	b	LBB0_44
LBB0_44:                                ;   in Loop: Header=BB0_39 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_39
LBB0_45:
	ldur	x0, [x29, #-24]
	bl	_free
	ldur	x0, [x29, #-32]
	bl	_free
	ldur	x0, [x29, #-16]
	bl	_free
	ldur	w0, [x29, #-8]
	ldp	x29, x30, [sp, #80]             ; 111-byte Folded Reload
	add	sp, sp, #96
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

