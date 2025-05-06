	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #648
	.cfi_def_cfa_offset 648
	stp	x29, x30, [sp, #648]            ; 16-byte Folded Spill
	add	x29, sp, #648
	.cfi_def_cfa w29, 16
	.cfi_offset w33, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-16]
	stur	x1, [x29, #-24]
	stur	wzr, [x29, #-28]
	ldur	x0, [x29, #-16]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #32]
	ldur	x0, [x29, #-24]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #28]
	mov	w8, #1
	str	w8, [sp, #20]
	str	wzr, [sp, #24]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #32]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_10
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #40
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #41
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #0
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #20]
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_1
LBB0_10:
	str	wzr, [sp, #24]
	b	LBB0_11
LBB0_11:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_20
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=1
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #40
	cset	w8, ne
	tbnz	w8, #0, LBB0_14
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_11 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_11 Depth=1
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #41
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_11 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_11 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #0
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_11 Depth=1
	str	wzr, [sp, #20]
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_11 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_11 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_11
LBB0_10:
	ldur	w8, [x29, #-28]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_13
	b	LBB0_11
LBB0_11:
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_13
	b	LBB0_12
LBB0_12:
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	stur	x8, [x29, #-8]
	b	LBB0_47
LBB0_13:
	stur	wzr, [x29, #-28]
	mov	w8, #1
	str	w8, [sp, #20]
	str	wzr, [sp, #24]
	b	LBB0_4
LBB0_4:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_33
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #40
	cset	w8, ne
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #41
	cset	w8, ne
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #0
	cset	w8, ge
	tbnz	w8, #0, LBB0_11
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_4 Depth=1
	ldr	w8, [sp, #20]
	str	wzr, [sp, #20]
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_4 Depth=1
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_4 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_4
LBB0_13:
	str	wzr, [sp, #24]
	b	LBB0_4
LBB0_4:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #32]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_43
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #40
	cset	w8, ne
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #41
	cset	w8, ne
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldur	w8, [x29, #-28]
	subs	w8, w8, #0
	cset	w8, ge
	tbnz	w8, #0, LBB0_11
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_4 Depth=1
	ldr	w8, [sp, #20]
	str	wzr, [sp, #20]
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_4 Depth=1
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_4 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_4
LBB0_13:
	ldur	w8, [x29, #-28]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_14
LBB0_14:
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	stur	x8, [x29, #-8]
	b	LBB0_47
LBB0_6:
	adrp	x8, l_.str.1@PAGE
	add	x8, x8, l_.str.1@PAGEOFF
	stur	x8, [x29, #-8]
	b	LBB0_7
LBB0_7:
	ldur	x0, [x29, #-8]
	ldp	x29, x33, [sp, #648]            ; 16-byte Folded Reload
	add	sp, sp, #648
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"Yes"

l_.str.1:                               ; @.str.1
	.asciz	"No"

.subsections_via_symbols

