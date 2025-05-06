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
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-8]
	str	w0, [sp, #20]
	str	w1, [sp, #16]
	str	x2, [sp, #8]
	str	wzr, [sp, #4]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_3
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	ldr	w10, [sp, #16]
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	add	w8, w8, #48
	ldrsw	x10, [sp, #4]
	mov	x9, x10
	add	w9, w9, #1
	str	w9, [sp, #4]
	sub	x9, x29, #24
	strb	w8, [x9, x10]
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #16]
	sdiv	w8, w8, w9
	stur	w8, [x29, #-28]
	b	LBB0_1
LBB0_3:
	str	wzr, [sp]
	b	LBB0_4
LBB0_4:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #4]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_4 Depth=1
	ldr	w8, [sp, #4]
	subs	w9, w8, #1
	str	w9, [sp, #4]
	sub	x8, x29, #24
	ldrb	w8, [x8]
	ldr	x9, [sp, #8]
	ldrsw	x10, [sp]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp]
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_4
LBB0_6:
	ldr	x8, [sp, #8]
	ldrsw	x9, [sp]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldur	x9, [x29, #-8]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:
	bl	___stack_chk_fail
LBB0_8:
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

