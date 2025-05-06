	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #368
	.cfi_def_cfa_offset 368
	stp	x28, x27, [sp, #336]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #356]            ; 16-byte Folded Spill
	add	x29, sp, #356
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w27, -24
	.cfi_offset w28, -32
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-24]
	str	x0, [sp, #64]
	str	w1, [sp, #60]
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	str	x8, [sp, #48]
	str	wzr, [sp, #44]
	str	wzr, [sp, #40]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #40]
	ldr	w9, [sp, #60]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_14
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	add	x0, sp, #72
	mov	w1, #0
	mov	x2, #256
	bl	_memset
	str	wzr, [sp, #36]
	str	wzr, [sp, #32]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [sp, #64]
	ldrsw	x9, [sp, #40]
	ldr	x8, [x8, x9, lsl #3]
	ldrsw	x9, [sp, #31]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #64]
	ldrsw	x9, [sp, #40]
	ldr	x8, [x8, x9, lsl #3]
	ldrsw	x9, [sp, #31]
	ldrsb	w1, [x8, x9]
	add	x0, sp, #72
	bl	_strchr
	subs	x8, x0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	add	x0, sp, #72
	str	x0, [sp, #16]                   ; 8-byte Folded Spill
	bl	_strlen
	ldr	x8, [sp, #16]                   ; 8-byte Folded Reload
	mov	x9, x0
	str	w9, [sp, #28]
	ldr	x9, [sp, #64]
	ldrsw	x10, [sp, #40]
	ldr	x9, [x9, x11, lsl #3]
	ldrsw	x10, [sp, #31]
	add	x9, x9, x10
	ldrb	w9, [x9]
	ldrsw	x11, [sp, #28]
	mov	x10, x8
	add	x10, x10, x11
	strb	w9, [x10]
	ldr	w9, [sp, #28]
	add	w9, w9, #1
	add	x8, x8, w9, sxtw
	strb	wzr, [x8]
	ldr	w8, [sp, #36]
	add	w8, w8, #1
	str	w8, [sp, #36]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #31]
	add	w8, w8, #1
	str	w8, [sp, #31]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #36]
	ldr	w9, [sp, #44]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_11
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #36]
	ldr	w9, [sp, #44]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_12
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #64]
	ldrsw	x9, [sp, #40]
	ldr	x0, [x8, x9, lsl #3]
	ldr	x1, [sp, #48]
	bl	_strcmp
	subs	w8, w0, #0
	cset	w8, ge
	tbnz	w8, #0, LBB0_12
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #64]
	ldrsw	x9, [sp, #40]
	ldr	x8, [x8, x9, lsl #3]
	str	x8, [sp, #48]
	ldr	w8, [sp, #36]
	str	w8, [sp, #44]
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #40]
	add	w8, w8, #1
	str	w8, [sp, #40]
	b	LBB0_1
LBB0_14:
	ldr	x8, [sp, #48]
	str	x8, [sp, #8]                    ; 8-byte Folded Spill
	ldur	x9, [x29, #-24]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_16
	b	LBB0_15
LBB0_15:
	bl	___stack_chk_fail
LBB0_16:
	ldr	x0, [sp, #8]                    ; 8-byte Folded Reload
	ldp	x29, x30, [sp, #356]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #333]            ; 16-byte Folded Reload
	add	sp, sp, #368
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.space	1

.subsections_via_symbols

