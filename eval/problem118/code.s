	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #96
	.cfi_def_cfa_offset 96
	stp	x29, x30, [sp, #80]             ; 16-byte Folded Spill
	add	x29, sp, #80
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	stur	w1, [x29, #-12]
	stur	x2, [x29, #-24]
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	stur	x8, [x29, #-32]
                                        ; kill: def $x8 killed $xzr
	str	xzr, [sp, #40]
	str	wzr, [sp, #36]
	str	wzr, [sp, #32]
	str	wzr, [sp, #28]
	ldur	x0, [x29, #-8]
	bl	_strlen
	str	x0, [sp, #16]
	ldr	x8, [sp, #16]
	add	x0, x8, #1
	bl	_malloc
	str	x0, [sp, #8]
	str	wzr, [sp, #4]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldrsw	x8, [sp, #4]
	ldr	x9, [sp, #16]
	subs	x8, x8, x9
	cset	w8, hi
	tbnz	w8, #0, LBB0_13
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #4]
	ldrsb	w0, [x8, x9]
	bl	_isspace
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #4]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_7
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #36]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #8]
	ldr	w9, [sp, #4]
	ldr	w10, [sp, #28]
	subs	w9, w9, w10
	add	x8, x8, w9, sxtw
	strb	wzr, [x8]
	ldr	x0, [sp, #40]
	ldr	w8, [sp, #32]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	lsl	x1, x8, #3
	bl	_realloc
	str	x0, [sp, #40]
	ldr	x0, [sp, #8]
	bl	_strlen
	add	x0, x0, #1
	bl	_malloc
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #32]
	str	x0, [x8, x9, lsl #3]
	ldr	x8, [sp, #400]
	ldrsw	x9, [sp, #32]
	ldr	x0, [x8, x9, lsl #3]
	ldr	x1, [sp, #8]
	mov	x2, #-1
	bl	___strcpy_chk
	ldr	w8, [sp, #32]
	add	w8, w8, #1
	str	w8, [sp, #32]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	stur	w8, [x29, #-12]
	str	wzr, [sp, #36]
	b	LBB0_11
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #4]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldr	x9, [sp, #8]
	ldr	w10, [sp, #4]
	ldr	w11, [sp, #28]
	subs	w10, w10, w11
	add	x9, x9, w10, sxtw
	strb	w8, [x9]
	ldur	x0, [x29, #-32]
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #4]
	ldrsb	w1, [x8, x9]
	bl	_strchr
	subs	x8, x0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #4]
	ldrb	w0, [x8, x9]
	bl	_isalpha
	subs	w8, w0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #36]
	add	w8, w8, #1
	str	w8, [sp, #36]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_1
LBB0_13:
	ldr	x0, [sp, #8]
	bl	_free
	ldr	w8, [sp, #32]
	ldur	x9, [x29, #-24]
	str	w8, [x9]
	ldr	x0, [sp, #40]
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	add	sp, sp, #96
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"aeiouAEIOU"

.subsections_via_symbols

