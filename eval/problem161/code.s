	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #80
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #648]            ; 16-byte Folded Spill
	add	x29, sp, #648
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	stur	x1, [x29, #-16]
	stur	w2, [x29, #-20]
	stur	w3, [x29, #-24]
	ldursw	x8, [x29, #-24]
	lsl	x0, x8, #2
	bl	_malloc
	stur	x0, [x29, #-32]
	ldursw	x8, [x29, #-24]
	lsl	x0, x8, #2
	bl	_malloc
	stur	x0, [x29, #-40]
	stur	wzr, [x29, #-44]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-44]
	ldur	w9, [x29, #-24]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	ldursw	x9, [x29, #-44]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-32]
	ldursw	x10, [x29, #-44]
	str	w8, [x9, x10, lsl #2]
	ldur	w8, [x29, #-44]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-44]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-44]
	add	w8, w8, #1
	stur	w8, [x29, #-44]
	b	LBB0_1
LBB0_4:
	stur	wzr, [x29, #-48]
	b	LBB0_5
LBB0_5:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;     Child Loop BB0_11 Depth 2
	ldur	w8, [x29, #-48]
	ldur	w9, [x29, #-20]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_16
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-48]
	ldr	x0, [x8, x9, lsl #3]
	adrp	x1, l_.str@PAGE
	add	x1, x1, l_.str@PAGEOFF
	bl	_strcmp
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_14
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_5 Depth=1
	b	LBB0_8
LBB0_8:                                 ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-48]
	ldrsw	x9, [x9, x10, lsl #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-48]
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, eq
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_8 Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-48]
	ldrsw	x9, [x9, x10, lsl #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-48]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_8
LBB0_10:                                ;   in Loop: Header=BB0_5 Depth=1
	b	LBB0_11
LBB0_11:                                ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-48]
	add	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-48]
	add	w10, w10, #1
	ldr	w9, [x9, w10, sxtw #2]
	subs	w8, w8, w9
	cset	w8, eq
	tbnz	w8, #0, LBB0_13
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-48]
	add	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-48]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_11
LBB0_13:                                ;   in Loop: Header=BB0_5 Depth=1
	ldur	x8, [x29, #-32]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-48]
	ldrsw	x9, [x9, x10, lsl #2]
	ldr	s1, [x8, x9, lsl #2]
                                        ; implicit-def: $d0
	fmov	s0, s1
	sshll.2d	v0, v0, #0
                                        ; kill: def $d0 killed $d0 killed $q0
	scvtf	d0, d0
	ldur	x8, [x29, #-32]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-48]
	add	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	ldr	s2, [x8, x9, lsl #2]
                                        ; implicit-def: $d1
	fmov	s1, s2
	sshll.2d	v1, v1, #0
                                        ; kill: def $d1 killed $d1 killed $q1
	scvtf	d1, d1
	bl	_pow
	fcvtzs	w8, d0
	ldur	x9, [x29, #-32]
	ldur	x10, [x29, #-40]
	ldursw	x11, [x29, #-48]
	ldrsw	x10, [x10, x11, lsl #2]
	str	w8, [x9, x10, lsl #2]
	ldur	x8, [x29, #-40]
	ldursw	x9, [x29, #-48]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-48]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_5 Depth=1
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_5 Depth=1
	ldur	w8, [x29, #-48]
	add	w8, w8, #1
	stur	w8, [x29, #-48]
	b	LBB0_5
LBB0_16:
	stur	wzr, [x29, #-52]
	b	LBB0_17
LBB0_17:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
                                        ;     Child Loop BB0_24 Depth 2
	ldur	w8, [x29, #-52]
	ldur	w9, [x29, #-20]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_32
	b	LBB0_18
LBB0_18:                                ;   in Loop: Header=BB0_17 Depth=1
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-52]
	ldr	x0, [x8, x9, lsl #3]
	adrp	x1, l_.str.1@PAGE
	add	x1, x1, l_.str.1@PAGEOFF
	bl	_strcmp
	subs	w8, w0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_20
	b	LBB0_11
LBB0_19:                                ;   in Loop: Header=BB0_17 Depth=1
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-52]
	ldr	x0, [x8, x9, lsl #3]
	adrp	x1, l_.str.2@PAGE
	add	x1, x1, l_.str.2@PAGEOFF
	bl	_strcmp
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_30
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_17 Depth=1
	b	LBB0_21
LBB0_21:                                ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-52]
	ldrsw	x9, [x9, x10, lsl #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-52]
	ldr	w9, [x9, x11, lsl #2]
	subs	w8, w8, w9
	cset	w8, eq
	tbnz	w8, #0, LBB0_23
	b	LBB0_23
LBB0_22:                                ;   in Loop: Header=BB0_22 Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-52]
	ldrsw	x9, [x9, x11, lsl #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldursw	x10, [x29, #-52]
	str	w8, [x9, x11, lsl #2]
	b	LBB0_22
LBB0_23:                                ;   in Loop: Header=BB0_17 Depth=1
	b	LBB0_4
LBB0_4:                                ;   Parent Loop BB0_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-444]
	ldur	w10, [x29, #-52]
	add	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-52]
	add	w10, w10, #1
	ldr	w9, [x9, w10, sxtw #2]
	subs	w8, w8, w9
	cset	w8, eq
	tbnz	w8, #0, LBB0_4
	b	LBB0_5
LBB0_5:                                ;   in Loop: Header=BB0_4 Depth=2
	ldur	x8, [x29, #-40]
	ldur	x9, [x29, #-444]
	ldur	w10, [x29, #-52]
	add	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-444]
	ldur	w10, [x29, #-52]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_4
LBB0_6:                                ;   in Loop: Header=BB0_17 Depth=1
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-52]
	ldr	x0, [x8, x9, lsl #3]
	adrp	x1, l_.str.3@PAGE
	add	x1, x1, l_.str.3@PAGEOFF
	bl	_strcmp
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                ;   in Loop: Header=BB0_17 Depth=1
	ldur	x8, [x29, #-32]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-52]
	add	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	ldr	w10, [x8, x9, lsl #2]
	ldur	x8, [x29, #-32]
	ldur	x9, [x29, #-444]
	ldursw	x11, [x29, #-52]
	ldrsw	x9, [x9, x11, lsl #2]
	add	x9, x8, x9, lsl #2
	ldr	w8, [x9]
	add	w8, w8, w10
	str	w8, [x9]
	b	LBB0_9
LBB0_8:                                 ;   in Loop: Header=BB0_17 Depth=1
	ldur	x8, [x29, #-32]
	ldur	x9, [x29, #-40]
	ldur	w10, [x29, #-52]
	add	w10, w10, #1
	ldrsw	x9, [x9, w10, sxtw #2]
	ldr	w10, [x8, x9, lsl #2]
	ldur	x8, [x29, #-32]
	ldur	x9, [x29, #-444]
	ldursw	x10, [x29, #-52]
	ldrsw	x9, [x9, x10, lsl #2]
	add	x9, x8, x9, lsl #2
	ldr	w8, [x9]
	subs	w8, w8, w10
	str	w8, [x9]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_17 Depth=1
	ldur	x8, [x29, #-40]
	ldursw	x9, [x29, #-52]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-444]
	ldur	w10, [x29, #-52]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_17 Depth=1
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_17 Depth=1
	ldur	w8, [x29, #-52]
	add	w8, w8, #1
	stur	w8, [x29, #-52]
	b	LBB0_17
LBB0_12:
	ldur	x8, [x29, #-32]
	ldr	w8, [x8]
	stur	w8, [x29, #-56]
	ldur	x0, [x29, #-32]
	bl	_free
	ldur	x0, [x29, #-40]
	bl	_free
	ldur	w0, [x29, #-56]
	ldp	x29, x30, [sp, #648]            ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"**"

l_.str.1:                               ; @.str.1
	.asciz	"*"

l_.str.2:                               ; @.str.2
	.asciz	"//"

l_.str.3:                               ; @.str.3
	.asciz	"+"

l_.str.4:                               ; @.str.4
	.asciz	"-"

.subsections_via_symbols

