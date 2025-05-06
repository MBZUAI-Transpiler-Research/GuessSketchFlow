	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	str	x8, [sp, #16]
	str	wzr, [sp, #12]
	ldur	x0, [x29, #-8]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #8]
	str	wzr, [sp, #4]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_6
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x0, [sp, #16]
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #8]
	ldrsb	w1, [x8, x9]
	bl	_strchr
	subs	x8, x0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_1
LBB0_6:
	ldr	w8, [sp, #8]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_11
	b	LBB0_7
LBB0_7:
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #8]
	subs	w9, w9, #1
	ldrsb	w8, [x8, w9, sxtw]
	subs	w8, w8, #121
	cset	w8, eq
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #8]
	subs	w9, w9, #1
	ldrsb	w8, [x8, w9, sxtw]
	subs	w8, w8, #89
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_10
LBB0_10:
	ldr	w0, [sp, #12]
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"aeiouAEIOU"

.subsections_via_symbols

