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
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-8]
	str	x0, [sp, #32]
	str	w1, [sp, #28]
	stur	xzr, [x29, #-24]
	sturh	wzr, [x29, #-16]
	str	xzr, [sp, #40]
	strh	wzr, [sp, #48]
	str	wzr, [sp, #24]
	str	wzr, [sp, #20]
	str	wzr, [sp, #16]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #16]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_13
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #16]
	ldrsb	w0, [x8, x9]
	bl	_isdigit
	subs	w8, w0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_7
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #16]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #20]
	mov	x9, x10
	add	w9, w9, #1
	str	w9, [sp, #20]
	sub	x9, x29, #24
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_6
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #32]
	ldrsw	x9, [sp, #16]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldrsw	x10, [sp, #20]
	mov	x9, x10
	add	w9, w9, #1
	str	w9, [sp, #20]
	add	x9, sp, #40
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_11
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #24]
	str	wzr, [sp, #20]
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_1
LBB0_13:
	ldr	w8, [sp, #28]
	str	w8, [sp, #4]                    ; 4-byte Folded Spill
	sub	x0, x29, #24
	bl	_atoi
	ldr	w8, [sp, #4]                    ; 4-byte Folded Reload
	subs	w8, w8, w0
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	add	x0, sp, #40
	bl	_atoi
	ldr	w8, [sp, #8]                    ; 4-byte Folded Reload
	subs	w8, w8, w0
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	ldur	x9, [x29, #-8]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:
	bl	___stack_chk_fail
LBB0_5:
	ldr	w0, [sp, #12]                   ; 4-byte Folded Reload
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	add	sp, sp, #96
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

