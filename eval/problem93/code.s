	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	s0, [sp, #8]
	str	s1, [sp, #4]
	str	s2, [sp]
	ldr	s0, [sp, #8]
	frinta	s0, s0
	ldr	s1, [sp, #8]
	fcmp	s0, s1
	cset	w8, eq
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	str	wzr, [sp, #12]
	b	LBB0_11
LBB0_2:
	ldr	s0, [sp, #4]
	frinta	s0, s0
	ldr	s1, [sp, #4]
	fcmp	s0, s1
	cset	w8, eq
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:
	str	wzr, [sp, #12]
	b	LBB0_11
LBB0_4:
	ldr	s0, [sp]
	frinta	s0, s0
	ldr	s1, [sp]
	fcmp	s0, s1
	cset	w8, eq
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:
	str	wzr, [sp, #12]
	b	LBB0_11
LBB0_6:
	ldr	s0, [sp, #8]
	ldr	s1, [sp, #4]
	fadd	s0, s0, s1
	ldr	s1, [sp]
	fcmp	s0, s1
	cset	w8, eq
	tbnz	w8, #0, LBB0_9
	b	LBB0_7
LBB0_7:
	ldr	s0, [sp, #8]
	ldr	s1, [sp]
	fadd	s0, s0, s1
	ldr	s1, [sp, #4]
	fcmp	s0, s1
	cset	w8, eq
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:
	ldr	s0, [sp, #4]
	ldr	s1, [sp]
	fadd	s0, s0, s1
	ldr	s1, [sp, #8]
	fcmp	s0, s1
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	mov	w8, #1
	str	w8, [sp, #12]
	b	LBB0_11
LBB0_10:
	str	wzr, [sp, #12]
	b	LBB0_11
LBB0_11:
	ldr	w0, [sp, #11]
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

