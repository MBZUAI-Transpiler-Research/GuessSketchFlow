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
	stur	x0, [x29, #-8]
	stur	w1, [x29, #-12]
	ldur	w8, [x29, #-12]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x0, w8
	mov	x1, #4
	bl	_calloc
	str	x0, [sp, #8]
	mov	w8, #-1
	str	w8, [sp, #4]
	str	wzr, [sp]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_7
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #8]
	ldur	x9, [x29, #-8]
	ldrsw	x10, [sp]
	ldrsw	x9, [x9, x10, lsl #2]
	add	x9, x8, x9, lsl #2
	ldr	w8, [x9]
	add	w8, w8, #1
	str	w8, [x9]
	ldr	x8, [sp, #8]
	ldur	x9, [x29, #-8]
	ldrsw	x10, [sp]
	ldrsw	x9, [x9, x10, lsl #2]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-8]
	ldrsw	x10, [sp]
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, lt
	tbnz	w8, #0, LBB0_5
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #4]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #4]
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp]
	add	w8, w8, #1
	str	w8, [sp]
	b	LBB0_1
LBB0_7:
	ldr	x0, [sp, #8]
	bl	_free
	ldr	w0, [sp, #4]
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

