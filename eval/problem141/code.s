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
	stur	x0, [x29, #-8]
	stur	x1, [x29, #-16]
	stur	wzr, [x29, #-20]
	str	wzr, [sp, #24]
	str	wzr, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldrsw	x8, [sp, #20]
	str	x8, [sp, #8]                    ; 8-byte Folded Spill
	ldur	x0, [x29, #-8]
	bl	_strlen
	ldr	x8, [sp, #8]                    ; 8-byte Folded Reload
	subs	x8, x8, x0
	cset	w8, hs
	tbnz	w8, #0, LBB0_13
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #20]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #32
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-20]
	add	w8, w8, #1
	stur	w8, [x29, #-20]
	b	LBB0_11
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-20]
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #24]
	add	x9, x8, x9
	mov	w8, #95
	strb	w8, [x9]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-20]
	subs	w8, w8, #2
	cset	w8, ne
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	mov	x10, x9
	add	w11, w11, #1
	str	w11, [sp, #24]
	add	x9, x8, x9
	mov	w8, #95
	strb	w8, [x9]
	ldur	x9, [x29, #-16]
	ldrsw	x10, [sp, #24]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp, #24]
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-20]
	subs	w8, w8, #2
	cset	w8, le
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #24]
	add	x9, x8, x9
	mov	w8, #45
	strb	w8, [x9]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	stur	wzr, [x29, #-20]
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #20]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldur	x9, [x29, #-16]
	ldrsw	x10, [sp, #24]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp, #24]
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_1
LBB0_13:
	ldur	w8, [x29, #-20]
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_15
	b	LBB0_14
LBB0_14:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #24]
	add	x9, x8, x9
	mov	w8, #95
	strb	w8, [x9]
	b	LBB0_15
LBB0_15:
	ldur	w8, [x29, #-20]
	subs	w8, w8, #2
	cset	w8, ne
	tbnz	w8, #0, LBB0_17
	b	LBB0_16
LBB0_16:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #24]
	add	x9, x8, x9
	mov	w8, #95
	strb	w8, [x9]
	ldur	x9, [x29, #-16]
	ldrsw	x10, [sp, #24]
	mov	x11, x11
	add	w11, w11, #1
	str	w11, [sp, #24]
	add	x9, x9, x11
	strb	w8, [x9]
	b	LBB0_17
LBB0_17:
	ldur	w8, [x29, #-20]
	subs	w8, w8, #2
	cset	w8, le
	tbnz	w8, #0, LBB0_19
	b	LBB0_18
LBB0_18:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	mov	x10, x9
	add	w10, w10, #1
	str	w11, [sp, #24]
	add	x9, x8, x9
	mov	w8, #45
	strb	w8, [x9]
	b	LBB0_19
LBB0_19:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

