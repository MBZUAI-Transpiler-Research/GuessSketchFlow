	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, #40]
	str	w1, [sp, #36]
	str	x2, [sp, #24]
	str	wzr, [sp, #20]
	mov	w8, #1
	str	w8, [sp, #44]
	str	wzr, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #36]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #12]
	ldr	w9, [x8, x9, lsl #2]
	ldr	w8, [sp, #20]
	add	w8, w8, w9
	str	w8, [sp, #20]
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #12]
	ldr	w9, [x8, x9, lsl #2]
	ldr	w8, [sp, #44]
	mul	w8, w8, w9
	str	w8, [sp, #44]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_4:
	ldr	w8, [sp, #20]
	ldr	x9, [sp, #24]
	str	w8, [x9]
	ldr	w8, [sp, #44]
	ldr	x9, [sp, #24]
	str	w8, [x9, #4]
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

