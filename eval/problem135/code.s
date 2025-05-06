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
	str	x0, [sp, #16]
	ldr	x0, [sp, #16]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #12]
	ldr	w8, [sp, #12]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	stur	wzr, [x29, #-4]
	b	LBB0_9
LBB0_2:
	ldr	x8, [sp, #16]
	ldr	w9, [sp, #12]
	subs	w9, w9, #1
	add	x8, x8, w9, sxtw
	ldrb	w8, [x8]
	strb	w8, [sp, #11]
	ldrb	w0, [sp, #11]
	bl	_isalpha
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:
	stur	wzr, [x29, #-4]
	b	LBB0_9
LBB0_4:
	ldr	w8, [sp, #12]
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:
	mov	w8, #1
	stur	w8, [x29, #-4]
	b	LBB0_9
LBB0_6:
	ldr	x8, [sp, #16]
	ldr	w9, [sp, #12]
	subs	w9, w9, #2
	add	x8, x8, w9, sxtw
	ldrb	w8, [x8]
	sturb	w8, [x29, #1]
	ldurb	w0, [x29, #1]
	bl	_isalpha
	subs	w8, w0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:
	stur	wzr, [x29, #-4]
	b	LBB0_9
LBB0_8:
	mov	w8, #1
	stur	w8, [x29, #-4]
	b	LBB0_9
LBB0_9:
	ldur	w0, [x29, #-4]
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

