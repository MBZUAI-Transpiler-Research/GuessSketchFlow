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
	stur	x0, [x29, #-8]
	stur	x1, [x29, #-16]
	ldur	x0, [x29, #-8]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-20]
	ldursw	x8, [x29, #-20]
	lsl	x0, x8, #3
	bl	_malloc
	stur	x0, [x29, #-2]
	ldur	w8, [x29, #-20]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x0, w8
	bl	_malloc
	str	x0, [sp, #24]
	ldr	x8, [sp, #24]
	strb	wzr, [x8]
	stur	wzr, [x29, #-4]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-4]
	ldur	w9, [x29, #-20]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x0, [sp, #24]
	bl	_strlen
	str	x0, [sp, #8]
	ldr	x0, [sp, #24]
	ldr	x8, [sp, #8]
	add	x1, x8, #2
	bl	_realloc
	str	x0, [sp, #24]
	ldur	x8, [x29, #-8]
	ldursw	x9, [x29, #-4]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldr	x9, [sp, #24]
	ldr	x10, [sp, #8]
	add	x9, x9, x10
	strb	w8, [x9]
	ldr	x8, [sp, #24]
	ldr	x9, [sp, #8]
	add	x9, x9, #1
	add	x8, x8, x9
	strb	wzr, [x8]
	ldr	x0, [sp, #24]
	bl	_strlen
	add	x0, x0, #1
	bl	_malloc
	ldur	x8, [x29, #-32]
	ldursw	x9, [x29, #-4]
	str	x0, [x8, x9, lsl #3]
	ldur	x8, [x29, #-32]
	ldursw	x9, [x29, #-4]
	ldr	x0, [x8, x9, lsl #3]
	ldr	x1, [sp, #24]
	mov	x2, #-1
	bl	___strcpy_chk
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-4]
	add	w8, w8, #1
	stur	w8, [x29, #-4]
	b	LBB0_1
LBB0_4:
	ldr	x0, [sp, #24]
	bl	_free
	ldur	w8, [x29, #-20]
	ldur	x9, [x29, #-16]
	str	w8, [x9]
	ldur	x0, [x29, #-32]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

