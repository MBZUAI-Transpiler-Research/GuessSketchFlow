	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #160
	.cfi_def_cfa_offset 160
	stp	x29, x30, [sp, #144]            ; 16-byte Folded Spill
	add	x29, sp, #144
	.cfi_def_cfa w29, 16
	.cfi_offset w33, -8
	.cfi_offset w29, -16
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-8]
	str	x0, [sp, #48]
	str	w1, [sp, #44]
	str	x2, [sp, #32]
	str	x3, [sp, #24]
	add	x0, sp, #56
	adrp	x1, l___const.func0.names@PAGE
	add	x1, x1, l___const.func0.names@PAGEOFF
	mov	x2, #80
	bl	_memcpy
	str	wzr, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #22]
	ldr	w9, [sp, #44]
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_11
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #16]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #44]
	ldr	w10, [sp, #22]
	subs	w9, w9, w10
	subs	w9, w9, #1
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	ldr	x9, [sp, #48]
	ldr	w10, [sp, #16]
	add	w10, w10, #1
	ldr	w9, [x9, w10, sxtw #2]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #12]
	ldr	x8, [sp, #48]
	ldr	w9, [sp, #16]
	add	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	ldr	x9, [sp, #48]
	ldrsw	x10, [sp, #16]
	str	w8, [x9, x10, lsl #2]
	ldr	w8, [sp, #12]
	ldr	x9, [sp, #48]
	ldr	w10, [sp, #16]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #22]
	add	w8, w8, #1
	str	w8, [sp, #22]
	b	LBB0_1
LBB0_100:
	ldr	x8, [sp, #24]
	str	wzr, [x8]
	ldr	w8, [sp, #44]
	subs	w8, w8, #1
	str	w8, [sp, #22]
	b	LBB0_11
LBB0_11:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #22]
	subs	w8, w8, #0
	cset	w8, lt
	tbnz	w8, #0, LBB0_17
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	subs	w8, w8, #1
	cset	w8, lt
	tbnz	w8, #0, LBB0_15
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	subs	w8, w8, #9
	cset	w8, gt
	tbnz	w8, #0, LBB0_5
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	x9, [sp, #24]
	ldr	w8, [x9]
	add	w8, w8, #1
	str	w8, [x9]
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_11 Depth=1
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_11 Depth=1
	ldr	w8, [sp, #22]
	subs	w8, w8, #1
	str	w8, [sp, #22]
	b	LBB0_11
LBB0_17:
	ldr	x8, [sp, #24]
	ldrsw	x9, [x8]
	mov	x8, #8
	mul	x0, x8, x9
	bl	_malloc
	ldr	x8, [sp, #32]
	str	x0, [x8]
	ldr	w8, [sp, #44]
	subs	w8, w8, #1
	str	w8, [sp, #22]
	str	wzr, [sp, #16]
	b	LBB0_18
LBB0_18:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #22]
	subs	w8, w8, #0
	cset	w8, lt
	mov	w9, #0
	str	w9, [sp, #8]                    ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_20
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	w8, [sp, #16]
	ldr	x9, [sp, #24]
	ldr	w9, [x9]
	subs	w8, w8, w9
	cset	w8, lt
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	w8, [sp, #8]                    ; 4-byte Folded Reload
	tbz	w8, #0, LBB0_26
	b	LBB0_21
LBB0_21:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	subs	w8, w8, #1
	cset	w8, lt
	tbnz	w8, #0, LBB0_25
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #22]
	ldr	w8, [x8, x9, lsl #2]
	subs	w8, w8, #9
	cset	w8, gt
	tbnz	w8, #0, LBB0_25
	b	LBB0_24
LBB0_24:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #22]
	ldrsw	x9, [x8, x9, lsl #2]
	add	x8, sp, #56
	ldr	x8, [x8, x9, lsl #3]
	ldr	x9, [sp, #32]
	ldr	x9, [x9]
	ldrsw	x10, [sp, #16]
	mov	x11, x11
	add	w11, w11, #1
	str	w11, [sp, #16]
	str	x8, [x9, x10, lsl #3]
	b	LBB0_25
LBB0_25:                                ;   in Loop: Header=BB0_18 Depth=1
	b	LBB0_26
LBB0_26:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	w8, [sp, #22]
	subs	w8, w8, #1
	str	w8, [sp, #22]
	b	LBB0_18
LBB0_27:
	ldur	x9, [x29, #-8]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_29
	b	LBB0_28
LBB0_28:
	bl	___stack_chk_fail
LBB0_29:
	ldp	x29, x33, [sp, #144]            ; 16-byte Folded Reload
	add	sp, sp, #160
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.space	1

l_.str.1:                               ; @.str.1
	.asciz	"One"

l_.str.2:                               ; @.str.2
	.asciz	"Two"

l_.str.3:                               ; @.str.3
	.asciz	"Three"

l_.str.4:                               ; @.str.4
	.asciz	"Four"

l_.str.5:                               ; @.str.5
	.asciz	"Five"

l_.str.6:                               ; @.str.6
	.asciz	"Six"

l_.str.7:                               ; @.str.7
	.asciz	"Seven"

l_.str.8:                               ; @.str.8
	.asciz	"Eight"

l_.str.9:                               ; @.str.9
	.asciz	"Nine"

	.section	__DATA,__const
	.p2align	3                               ; @__const.func0.names
l___const.func0.names:
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

.subsections_via_symbols

