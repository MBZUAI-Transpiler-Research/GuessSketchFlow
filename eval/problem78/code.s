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
	stur	w0, [x29, #-8]
	stur	wzr, [x29, #-12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-12]
	ldur	w9, [x29, #-12]
	mul	w8, w8, w9
	ldur	w9, [x29, #-12]
	mul	w8, w8, w9
	str	w8, [sp, #16]                   ; 4-byte Folded Spill
	ldur	w0, [x29, #-8]
	bl	_abs
	ldr	w8, [sp, #16]                   ; 4-byte Folded Reload
	subs	w8, w8, w0
	cset	w8, gt
	tbnz	w8, #0, LBB0_6
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-12]
	ldur	w9, [x29, #-12]
	mul	w8, w8, w9
	ldur	w9, [x29, #-12]
	mul	w8, w8, w9
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	ldur	w0, [x29, #-8]
	bl	_abs
	ldr	w8, [sp, #12]                   ; 4-byte Folded Reload
	subs	w8, w8, w0
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:
	mov	w8, #1
	stur	w8, [x29, #-4]
	b	LBB0_7
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-12]
	add	w8, w8, #1
	stur	w8, [x29, #-12]
	b	LBB0_1
LBB0_6:
	stur	wzr, [x29, #-4]
	b	LBB0_7
LBB0_7:
	ldur	w0, [x29, #-4]
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

