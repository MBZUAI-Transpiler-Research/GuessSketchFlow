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
	stur	w0, [x29, #-4]
	str	x1, [sp, #16]
	str	wzr, [sp, #12]
	str	wzr, [sp, #8]
	ldur	w0, [x29, #-4]
	bl	_abs
	stur	w0, [x29, #-4]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-4]
	mov	w10, #10
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	str	w8, [sp, #4]
	ldr	w8, [sp, #4]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_3
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_4
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-4]
	mov	w9, #10
	sdiv	w8, w8, w9
	stur	w8, [x29, #-4]
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-4]
	subs	w8, w8, #0
	cset	w8, gt
	tbnz	w8, #0, LBB0_1
	b	LBB0_6
LBB0_6:
	ldr	w8, [sp, #12]
	ldr	x9, [sp, #16]
	str	w8, [x9]
	ldr	w8, [sp, #8]
	ldr	x9, [sp, #16]
	str	w8, [x9, #4]
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

