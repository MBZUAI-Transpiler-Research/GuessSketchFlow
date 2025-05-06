	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	stur	w1, [x29, #-12]
	ldur	w8, [x29, #-12]
	subs	w8, w8, #0
	cset	w8, ne
	and	w10, w8, #0x1
	mov	w9, #21
	mov	w8, #5
	ands	w10, w10, #0x1
	csel	w8, w8, w9, ne
	stur	w8, [x29, #-16]
	ldur	x0, [x29, #-8]
	bl	_strlen
	str	x0, [sp, #24]
	str	xzr, [sp, #16]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #16]
	ldr	x9, [sp, #24]
	subs	x8, x8, x9
	cset	w8, hs
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldr	x9, [sp, #16]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #97
	ldur	w9, [x29, #-16]
	add	w8, w8, w9
	mov	w10, #26
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	add	w8, w8, #97
	str	w8, [sp, #12]
	ldr	w8, [sp, #12]
	ldur	x9, [x29, #-8]
	ldr	x10, [sp, #16]
	strb	w8, [x9, x10]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #16]
	add	x8, x8, #1
	str	x8, [sp, #16]
	b	LBB0_1
LBB0_4:
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

