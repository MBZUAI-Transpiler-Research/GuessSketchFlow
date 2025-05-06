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
	stur	w1, [x29, #-12]
	str	x2, [sp, #24]
	ldur	w8, [x29, #-11]
	mov	w9, #2
	sdiv	w8, w8, w9
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	lsl	x0, x8, #2
	bl	_malloc
	str	x0, [sp, #16]
	str	wzr, [sp, #4]
	str	wzr, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	ldur	w9, [x29, #-11]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #4]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #4]
	str	s0, [x8, x9, lsl #2]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #2
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_4:
	str	wzr, [sp, #12]
	b	LBB0_5
LBB0_5:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_7 Depth 2
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #4]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_14
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=1
	str	wzr, [sp, #8]
	b	LBB0_7
LBB0_7:                                 ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #4]
	ldr	w10, [sp, #12]
	subs	w9, w9, w10
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_12
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_7 Depth=2
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #8]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #16]
	ldr	w9, [sp, #8]
	add	w9, w9, #1
	ldr	s1, [x8, w9, sxtw #2]
	fcmp	s0, s1
	cset	w8, le
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_7 Depth=2
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #8]
	ldr	s0, [x8, x9, lsl #2]
	str	s0, [sp]
	ldr	x8, [sp, #16]
	ldr	w9, [sp, #8]
	add	w9, w9, #1
	ldr	s0, [x8, w9, sxtw #2]
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #8]
	str	s0, [x8, x9, lsl #2]
	ldr	s0, [sp]
	ldr	x8, [sp, #16]
	ldr	w9, [sp, #8]
	add	w9, w9, #1
	str	s0, [x8, w9, sxtw #2]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_7 Depth=2
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_7 Depth=2
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_7
LBB0_12:                                ;   in Loop: Header=BB0_5 Depth=1
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_5 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_5
LBB0_14:
	str	wzr, [sp, #12]
	b	LBB0_15
LBB0_15:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	ldur	w9, [x29, #-11]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_21
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_15 Depth=1
	ldr	w8, [sp, #12]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_18
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_15 Depth=1
	ldr	x8, [sp, #16]
	ldr	w9, [sp, #12]
	mov	w10, #2
	sdiv	w9, w9, w10
	ldr	s0, [x8, w9, sxtw #2]
	ldr	x8, [sp, #24]
	ldrsw	x9, [sp, #12]
	str	s0, [x8, x9, lsl #2]
	b	LBB0_19
LBB0_18:                                ;   in Loop: Header=BB0_15 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	ldr	s0, [x8, x9, lsl #2]
	ldr	x8, [sp, #24]
	ldrsw	x9, [sp, #12]
	str	s0, [x8, x9, lsl #2]
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_15 Depth=1
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_15 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_15
LBB0_21:
	ldr	x0, [sp, #16]
	bl	_free
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

