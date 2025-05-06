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
	ldur	x0, [x29, #-8]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-16]
	ldur	w8, [x29, #-16]
	add	w8, w8, #2
	mov	w9, #3
	sdiv	w8, w8, w9
	stur	w8, [x29, #-20]
	str	wzr, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #20]
	ldur	w9, [x29, #-20]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_12
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	mov	w9, #3
	mul	w8, w8, w9
	add	w8, w8, #3
	ldur	w9, [x29, #-16]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #3
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	b	LBB0_5
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-16]
	ldr	w9, [sp, #20]
	mov	w10, #3
	mul	w9, w9, w10
	subs	w8, w8, w9
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]                    ; 4-byte Folded Reload
	stur	w8, [x29, #-2]
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #20]
	mov	w10, #3
	mul	w9, w9, w10
	add	x1, x8, w9, sxtw
	ldursw	x2, [x29, #-2]
	add	x0, sp, #24
	str	x0, [sp]                        ; 8-byte Folded Spill
	mov	x3, #4
	bl	___strncpy_chk
	ldr	x8, [sp]                        ; 8-byte Folded Reload
	ldursw	x9, [x29, #-2]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldur	w8, [x29, #-2]
	subs	w8, w8, #3
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-12]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrb	w8, [sp, #26]
	strb	w8, [sp, #19]
	ldrb	w8, [sp, #25]
	sturb	w8, [x29, #-2]
	ldrb	w8, [sp, #24]
	strb	w8, [sp, #25]
	ldrb	w8, [sp, #19]
	strb	w8, [sp, #24]
	b	LBB0_9
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrb	w8, [sp, #24]
	strb	w8, [sp, #18]
	ldrb	w8, [sp, #25]
	strb	w8, [sp, #24]
	ldrb	w8, [sp, #26]
	strb	w8, [sp, #25]
	ldrb	w8, [sp, #18]
	sturb	w8, [x29, #-2]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #20]
	mov	w10, #3
	mul	w9, w9, w11
	add	x0, x8, w9, sxtw
	ldrsw	x2, [sp, #16]
	add	x1, sp, #24
	mov	x3, #-1
	bl	___strncpy_chk
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	add	w8, w8, #1
	str	w8, [sp, #20]
	b	LBB0_1
LBB0_12:
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

