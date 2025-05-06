	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, #32]
	str	w1, [sp, #28]
	str	x2, [sp, #16]
	str	w3, [sp, #12]
	str	wzr, [sp, #8]
	str	wzr, [sp, #4]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #4]
	ldr	w9, [sp, #28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_6
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #4]
	ldr	w8, [x8, x9, lsl #2]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_1
LBB0_6:
	str	wzr, [sp]
	b	LBB0_7
LBB0_7:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_12
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_7 Depth=1
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp]
	ldr	w8, [x8, x9, lsl #2]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_7 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_7 Depth=1
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_7 Depth=1
	ldr	w8, [sp]
	add	w8, w8, #1
	str	w8, [sp]
	b	LBB0_7
LBB0_12:
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #28]
	subs	w8, w8, w9
	cset	w8, lt
	tbnz	w8, #0, LBB0_14
	b	LBB0_13
LBB0_13:
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	str	x8, [sp, #40]
	b	LBB0_15
LBB0_14:
	adrp	x8, l_.str.1@PAGE
	add	x8, x8, l_.str.1@PAGEOFF
	str	x8, [sp, #40]
	b	LBB0_15
LBB0_15:
	ldr	x0, [sp, #40]
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"YES"

l_.str.1:                               ; @.str.1
	.asciz	"NO"

.subsections_via_symbols

