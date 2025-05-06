	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #304
	.cfi_def_cfa_offset 304
	stp	x28, x27, [sp, #272]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #288]            ; 16-byte Folded Spill
	add	x29, sp, #288
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w27, -24
	.cfi_offset w28, -32
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-24]
	str	x0, [sp, #48]
	str	x1, [sp, #40]
	str	wzr, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #20]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #47
	cset	w8, eq
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #20]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #20]
	sub	x9, x29, #124
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_1
LBB0_4:
	ldrsw	x9, [sp, #20]
	sub	x0, x29, #124
	mov	x8, x0
	add	x8, x8, x9
	strb	wzr, [x8]
	bl	_atoi
	str	w0, [sp, #36]
	str	wzr, [sp, #16]
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_5
LBB0_5:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #20]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #20]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #16]
	add	x9, sp, #64
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_5
LBB0_8:
	ldrsw	x9, [sp, #16]
	add	x0, sp, #64
	mov	x8, x0
	add	x8, x8, x9
	strb	wzr, [x8]
	bl	_atoi
	str	w0, [sp, #32]
	str	wzr, [sp, #20]
	b	LBB0_9
LBB0_9:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #20]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #47
	cset	w8, eq
	tbnz	w8, #0, LBB0_12
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_9 Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #20]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #20]
	sub	x9, x29, #124
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_9 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_9
LBB0_12:
	ldrsw	x9, [sp, #20]
	sub	x0, x29, #124
	mov	x8, x0
	add	x8, x8, x9
	strb	wzr, [x8]
	bl	_atoi
	str	w0, [sp, #28]
	str	wzr, [sp, #120]
	ldr	w8, [sp, #124]
	add	w8, w8, #1
	str	w8, [sp, #124]
	b	LBB0_13
LBB0_13:                                ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #20]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_16
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_13 Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #20]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #16]
	add	x9, sp, #64
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_13 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_13
LBB0_16:
	ldrsw	x9, [sp, #16]
	add	x0, sp, #64
	mov	x8, x0
	add	x8, x8, x9
	strb	wzr, [x8]
	bl	_atoi
	str	w0, [sp, #24]
	ldr	w8, [sp, #36]
	ldr	w9, [sp, #28]
	mul	w8, w8, w9
	ldr	w9, [sp, #32]
	ldr	w10, [sp, #24]
	mul	w10, w9, w10
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_18
	b	LBB0_17
LBB0_17:
	mov	w8, #1
	str	w8, [sp, #60]
	b	LBB0_19
LBB0_18:
	str	wzr, [sp, #60]
	b	LBB0_19
LBB0_19:
	ldr	w8, [sp, #60]
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	ldur	x9, [x29, #-24]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_21
	b	LBB0_20
LBB0_20:
	bl	___stack_chk_fail
LBB0_21:
	ldr	w0, [sp, #12]                   ; 4-byte Folded Reload
	ldp	x29, x30, [sp, #288]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #272]            ; 16-byte Folded Reload
	add	sp, sp, #304
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

