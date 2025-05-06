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
	stur	w0, [x29, #-4]
	ldur	w9, [x29, #-4]
                                        ; implicit-def: $x8
	mov	x8, x9
	mov	x9, sp
	str	x8, [x9]
	sub	x0, x29, #10
	mov	w1, #0
	mov	x2, #6
	adrp	x3, l_.str@PAGE
	add	x3, x3, l_.str@PAGEOFF
	bl	___sprintf_chk
	stur	wzr, [x29, #-16]
	stur	wzr, [x29, #-20]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldursw	x9, [x29, #-20]
	sub	x8, x29, #10
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldursw	x9, [x29, #-20]
	sub	x8, x29, #10
	ldrsb	w8, [x8, x9]
	subs	w9, w8, #48
	ldur	w8, [x29, #-16]
	add	w8, w8, w9
	stur	w8, [x29, #-16]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-20]
	add	w8, w8, #1
	stur	w8, [x29, #-20]
	b	LBB0_1
LBB0_4:
	mov	x0, #33
	bl	_malloc
	stur	x0, [x29, #-28]
	str	wzr, [sp, #28]
	ldur	w8, [x29, #-16]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:
	ldur	x8, [x29, #-28]
	ldrsw	x9, [sp, #28]
	mov	x10, x9
	add	w10, w10, #1
	str	w10, [sp, #28]
	add	x9, x8, x9
	mov	w8, #48
	strb	w8, [x9]
	b	LBB0_11
LBB0_6:
	b	LBB0_7
LBB0_7:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-16]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_7 Depth=1
	ldur	w8, [x29, #-16]
	mov	w9, #2
	sdiv	w10, w8, w9
	mul	w10, w10, w9
	subs	w8, w8, w10
	add	w8, w8, #48
	ldr	x10, [sp, #32]
	ldrsw	x11, [sp, #28]
	mov	x12, x11
	add	w12, w12, #1
	str	w12, [sp, #28]
	strb	w8, [x10, x11]
	ldur	w8, [x29, #-16]
	sdiv	w8, w8, w9
	stur	w8, [x29, #-16]
	b	LBB0_7
LBB0_9:
	b	LBB0_10
LBB0_10:
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #28]
	add	x8, x8, x9
	strb	wzr, [x8]
	str	wzr, [sp, #24]
	b	LBB0_11
LBB0_11:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #28]
	mov	w10, #2
	sdiv	w9, w9, w10
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_14
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #24]
	add	x8, x8, x9
	ldrb	w8, [x8]
	strb	w8, [sp, #23]
	ldr	x8, [sp, #32]
	ldr	w9, [sp, #28]
	ldr	w10, [sp, #24]
	subs	w9, w9, w10
	subs	w9, w9, #1
	add	x8, x8, w9, sxtw
	ldrb	w8, [x8]
	ldr	x9, [sp, #32]
	ldrsw	x10, [sp, #24]
	add	x9, x9, x10
	strb	w8, [x9]
	ldrb	w8, [sp, #23]
	ldr	x9, [sp, #32]
	ldr	w10, [sp, #28]
	ldr	w11, [sp, #24]
	subs	w10, w10, w11
	subs	w10, w10, #1
	add	x9, x9, w10, sxtw
	strb	w8, [x9]
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_11
LBB0_14:
	ldr	x0, [sp, #32]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"%d"

.subsections_via_symbols

