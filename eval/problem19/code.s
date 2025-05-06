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
	stur	x0, [x29, #-16]
	str	x1, [sp, #24]
	str	wzr, [sp, #20]
	ldur	x0, [x29, #-16]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #16]
	ldr	x0, [sp, #24]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #12]
	ldr	w8, [sp, #16]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	stur	wzr, [x29, #-4]
	b	LBB0_9
LBB0_2:
	str	wzr, [sp, #8]
	b	LBB0_3
LBB0_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #16]
	ldr	w10, [sp, #12]
	subs	w9, w9, w10
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #8]
	add	x0, x8, x9
	ldr	x1, [sp, #24]
	ldrsw	x2, [sp, #12]
	bl	_strncmp
	subs	w8, w0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=1
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_3
LBB0_8:
	ldr	w8, [sp, #20]
	stur	w8, [x29, #-4]
	b	LBB0_9
LBB0_9:
	ldur	w0, [x29, #-4]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

