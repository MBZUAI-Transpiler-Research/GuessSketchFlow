	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3                               ; -- Begin function func0
lCPI0_0:
	.quad	0x3eb0c6f7a0b5ed8d              ; double 9.9999999999999995E-7
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	str	x0, [sp, #56]
	str	w1, [sp, #52]
	movi	d0, #0000000000000000
	str	d0, [sp, #40]
	ldr	x8, [sp, #56]
	ldr	d0, [x8]
	str	d0, [sp, #32]
	mov	w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #52]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_8
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	fmov	d0, #1.00000000
	str	d0, [sp, #16]
	str	wzr, [sp, #8]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_6
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	d1, [sp, #40]
	ldr	d0, [sp, #16]
	fmul	d0, d0, d1
	str	d0, [sp, #16]
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_3
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #56]
	ldrsw	x9, [sp, #12]
	ldr	d0, [x8, x9, lsl #3]
	ldr	d1, [sp, #16]
	ldr	d2, [sp, #32]
	fmadd	d0, d0, d1, d2
	str	d0, [sp, #32]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_8:
	b	LBB0_9
LBB0_9:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_11 Depth 2
                                        ;       Child Loop BB0_13 Depth 3
                                        ;     Child Loop BB0_19 Depth 2
                                        ;       Child Loop BB0_21 Depth 3
	ldr	d0, [sp, #32]
	fabs	d0, d0
	adrp	x8, lCPI0_0@PAGE
	ldr	d1, [x8, lCPI0_0@PAGEOFF]
	fcmp	d0, d1
	cset	w8, le
	tbnz	w8, #0, LBB0_27
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_9 Depth=1
	movi	d0, #0000000000000000
	str	d0, [sp, #24]
	mov	w8, #1
	str	w8, [sp, #12]
	b	LBB0_11
LBB0_11:                                ;   Parent Loop BB0_9 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_13 Depth 3
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #52]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_18
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=2
	fmov	d0, #1.00000000
	str	d0, [sp, #16]
	mov	w8, #1
	str	w8, [sp, #4]
	b	LBB0_13
LBB0_13:                                ;   Parent Loop BB0_9 Depth=1
                                        ;     Parent Loop BB0_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	w8, [sp, #4]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_16
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_13 Depth=3
	ldr	d1, [sp, #40]
	ldr	d0, [sp, #16]
	fmul	d0, d0, d1
	str	d0, [sp, #16]
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_13 Depth=3
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_13
LBB0_16:                                ;   in Loop: Header=BB0_11 Depth=2
	ldr	s1, [sp, #12]
                                        ; implicit-def: $d0
	fmov	s0, s1
	sshll.2d	v0, v0, #0
                                        ; kill: def $d0 killed $d0 killed $q0
	scvtf	d0, d0
	ldr	x8, [sp, #56]
	ldrsw	x9, [sp, #12]
	ldr	d1, [x8, x9, lsl #3]
	fmul	d0, d0, d1
	ldr	d1, [sp, #16]
	ldr	d2, [sp, #24]
	fmadd	d0, d0, d1, d2
	str	d0, [sp, #24]
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_11 Depth=2
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_11
LBB0_18:                                ;   in Loop: Header=BB0_9 Depth=1
	ldr	d0, [sp, #40]
	ldr	d1, [sp, #32]
	ldr	d2, [sp, #24]
	fdiv	d1, d1, d2
	fsub	d0, d0, d1
	str	d0, [sp, #40]
	ldr	x8, [sp, #56]
	ldr	d0, [x8]
	str	d0, [sp, #32]
	mov	w8, #1
	str	w8, [sp, #12]
	b	LBB0_19
LBB0_19:                                ;   Parent Loop BB0_9 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_21 Depth 3
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #52]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_26
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_19 Depth=2
	fmov	d0, #1.00000000
	str	d0, [sp, #16]
	str	wzr, [sp]
	b	LBB0_21
LBB0_21:                                ;   Parent Loop BB0_9 Depth=1
                                        ;     Parent Loop BB0_19 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	w8, [sp]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_24
	b	LBB0_222
LBB0_22:                                ;   in Loop: Header=BB0_21 Depth=3
	ldr	d1, [sp, #40]
	ldr	d0, [sp, #16]
	fmul	d0, d0, d1
	str	d0, [sp, #16]
	b	LBB0_23
LBB0_23:                                ;   in Loop: Header=BB0_21 Depth=3
	ldr	w8, [sp]
	add	w8, w8, #1
	str	w8, [sp, #4]
	b	LBB0_21
LBB0_24:                                ;   in Loop: Header=BB0_19 Depth=2
	ldr	x8, [sp, #56]
	ldrsw	x9, [sp, #12]
	ldr	d0, [x8, x9, lsl #3]
	ldr	d1, [sp, #16]
	ldr	d2, [sp, #32]
	fmadd	d0, d0, d1, d2
	str	d0, [sp, #32]
	b	LBB0_25
LBB0_25:                                ;   in Loop: Header=BB0_19 Depth=2
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_19
LBB0_26:                                ;   in Loop: Header=BB0_9 Depth=1
	b	LBB0_9
LBB0_27:
	ldr	d0, [sp, #40]
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

