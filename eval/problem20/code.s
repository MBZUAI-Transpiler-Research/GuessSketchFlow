	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #176
	.cfi_def_cfa_offset 176
	stp	x29, x30, [sp, #160]            ; 16-byte Folded Spill
	add	x29, sp, #160
	.cfi_def_cfa w29, 16
	.cfi_offset w33, -8
	.cfi_offset w29, -16
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-8]
	str	x0, [sp, #32]
	stur	xzr, [x29, #-40]
	stur	xzr, [x29, #-32]
	stur	xzr, [x29, #-24]
	stur	xzr, [x29, #-16]
	add	x0, sp, #40
	adrp	x1, l___const.func0.numto@PAGE
	add	x1, x1, l___const.func0.numto@PAGEOFF
	mov	x2, #80
	bl	_memcpy
	str	wzr, [sp, #28]
	ldr	x8, [sp, #32]
	ldrb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_17
	b	LBB0_1
LBB0_1:
	b	LBB0_2
LBB0_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
                                        ;     Child Loop BB0_9 Depth 2
	str	wzr, [sp, #24]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #32
	cset	w8, eq
	mov	w9, #0
	str	w9, [sp]                        ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, ne
	str	w8, [sp]                        ; 4-byte Folded Spill
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp]                        ; 4-byte Folded Reload
	tbz	w8, #0, LBB0_8
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #24]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #24]
	add	x9, sp, #2
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_2 Depth=1
	ldrsw	x9, [sp, #24]
	add	x8, sp, #2
	add	x8, x8, x9
	strb	wzr, [x8]
	str	wzr, [sp, #20]
	b	LBB0_9
LBB0_9:                                 ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #20]
	subs	w8, w8, #10
	cset	w8, ge
	tbnz	w8, #0, LBB0_14
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_9 Depth=2
	ldrsw	x9, [sp, #20]
	add	x8, sp, #40
	ldr	x1, [x8, x9, lsl #3]
	add	x0, sp, #2
	bl	_strcmp
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_12
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_2 Depth=1
	ldrsw	x9, [sp, #20]
	sub	x8, x29, #44
	add	x9, x8, x9, lsl #2
	ldr	w8, [x9]
	add	w8, w8, #1
	str	w8, [x9]
	b	LBB0_14
LBB0_12:                                ;   in Loop: Header=BB0_9 Depth=2
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_9 Depth=2
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_9
LBB0_14:                                ;   in Loop: Header=BB0_2 Depth=1
	ldr	w8, [sp, #24]
	add	w9, w8, #1
	ldr	x8, [sp, #32]
	add	x8, x8, w9, sxtw
	str	x8, [sp, #32]
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_2 Depth=1
	ldr	x8, [sp, #32]
	subs	x8, x8, #1
	ldrb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_2
	b	LBB0_16
LBB0_16:
	b	LBB0_17
LBB0_17:
	str	wzr, [sp, #24]
	b	LBB0_18
LBB0_18:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_20 Depth 2
                                        ;       Child Loop BB0_222 Depth 3
	ldr	w8, [sp, #24]
	subs	w8, w8, #10
	cset	w8, ge
	tbnz	w8, #0, LBB0_29
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_18 Depth=1
	str	wzr, [sp, #20]
	b	LBB0_20
LBB0_20:                                ;   Parent Loop BB0_18 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_22 Depth 3
	ldr	w8, [sp, #20]
	ldrsw	x10, [sp, #24]
	sub	x9, x29, #44
	ldr	w9, [x9, x11, lsl #2]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_27
	b	LBB0_21
LBB0_21:                                ;   in Loop: Header=BB0_20 Depth=2
	str	wzr, [sp, #16]
	b	LBB0_22
LBB0_22:                                ;   Parent Loop BB0_18 Depth=1
                                        ;     Parent Loop BB0_20 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldrsw	x9, [sp, #24]
	add	x8, sp, #40
	ldr	x8, [x8, x9, lsl #3]
	ldrsw	x9, [sp, #16]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_25
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_22 Depth=3
	ldrsw	x9, [sp, #24]
	add	x8, sp, #40
	ldr	x8, [x8, x9, lsl #3]
	ldrsw	x9, [sp, #16]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #28]
	adrp	x9, _func0.out@PAGE
	add	x9, x9, _func0.out@PAGEOFF
	add	x9, x9, x11
	strb	w8, [x9]
	b	LBB0_24
LBB0_24:                                ;   in Loop: Header=BB0_22 Depth=3
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	ldr	w8, [sp, #28]
	add	w8, w8, #1
	str	w8, [sp, #28]
	b	LBB0_22
LBB0_5:                                ;   in Loop: Header=BB0_20 Depth=2
	ldrsw	x9, [sp, #28]
	mov	x8, x9
	add	w8, w8, #1
	str	w8, [sp, #28]
	adrp	x8, _func0.out@PAGE
	add	x8, x8, _func0.out@PAGEOFF
	add	x9, x8, x9
	mov	w8, #32
	strb	w8, [x9]
	b	LBB0_6
LBB0_6:                                ;   in Loop: Header=BB0_20 Depth=2
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_20
LBB0_7:                                ;   in Loop: Header=BB0_18 Depth=1
	b	LBB0_28
LBB0_8:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_18
LBB0_9:
	ldr	w8, [sp, #28]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_31
	b	LBB0_30
LBB0_30:
	ldr	w8, [sp, #28]
	subs	w9, w8, #1
	adrp	x8, _func0.out@PAGE
	add	x8, x8, _func0.out@PAGEOFF
	add	x8, x8, w9, sxtw
	strb	wzr, [x8]
	b	LBB0_32
LBB0_31:
	adrp	x8, _func0.out@PAGE
	strb	wzr, [x8, _func0.out@PAGEOFF]
	b	LBB0_32
LBB0_32:
	ldur	x9, [x29, #-8]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_34
	b	LBB0_33
LBB0_33:
	bl	___stack_chk_fail
LBB0_34:
	adrp	x0, _func0.out@PAGE
	add	x0, x0, _func0.out@PAGEOFF
	ldp	x29, x33, [sp, #160]            ; 16-byte Folded Reload
	add	sp, sp, #176
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"zero"

l_.str.1:                               ; @.str.1
	.asciz	"one"

l_.str.2:                               ; @.str.2
	.asciz	"two"

l_.str.3:                               ; @.str.3
	.asciz	"three"

l_.str.4:                               ; @.str.4
	.asciz	"four"

l_.str.5:                               ; @.str.5
	.asciz	"five"

l_.str.6:                               ; @.str.6
	.asciz	"six"

l_.str.7:                               ; @.str.7
	.asciz	"seven"

l_.str.8:                               ; @.str.8
	.asciz	"eight"

l_.str.9:                               ; @.str.9
	.asciz	"nine"

	.section	__DATA,__const
	.p2align	3                               ; @__const.func0.numto
l___const.func0.numto:
	.quad	l_.str
	.quad	l_.str.1
	.quad	l_.str.2
	.quad	l_.str.3
	.quad	l_.str.4
	.quad	l_.str.5
	.quad	l_.str.6
	.quad	l_.str.7
	.quad	l_.str.8
	.quad	l_.str.9

.zerofill __DATA,__bss,_func0.out,1100,0 ; @func0.out
.subsections_via_symbols

