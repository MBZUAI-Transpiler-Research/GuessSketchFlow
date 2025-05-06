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
	stur	x0, [x29, #-16]
	stur	x1, [x29, #-24]
	ldur	x0, [x29, #-16]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-28]
	ldur	x0, [x29, #-24]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-22]
	ldur	w8, [x29, #-28]
	ldur	w9, [x29, #-24]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	ldur	w8, [x29, #-28]
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	b	LBB0_3
LBB0_2:
	ldur	w8, [x29, #-24]
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	b	LBB0_3
LBB0_3:
	ldr	w8, [sp, #8]                    ; 4-byte Folded Reload
	str	w8, [sp, #28]
	ldr	w8, [sp, #28]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	lsr	x0, x8, #0
	bl	_malloc
	str	x0, [sp, #16]
	ldr	x8, [sp, #16]
	subs	x8, x8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:
                                        ; kill: def $x8 killed $xzr
	stur	xzr, [x29, #-8]
	b	LBB0_11
LBB0_5:
	stur	wzr, [x29, #-28]
	b	LBB0_6
LBB0_6:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-28]
	ldr	w9, [sp, #28]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_9
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_6 Depth=1
	ldur	x8, [x29, #-16]
	ldursw	x9, [x29, #-28]
	ldrsb	w8, [x8, x9]
	ldur	x9, [x29, #-24]
	ldursw	x10, [x29, #-28]
	ldrsb	w9, [x9, x10]
	subs	w8, w8, w9
	cset	w8, eq
	and	w10, w8, #0x1
	mov	w9, #49
	mov	w8, #48
	ands	w10, w10, #0x1
	csel	w8, w8, w9, ne
	ldr	x9, [sp, #16]
	ldursw	x10, [x29, #-28]
	strb	w8, [x9, x10]
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_6 Depth=1
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	stur	w8, [x29, #-28]
	b	LBB0_6
LBB0_9:
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #28]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldr	x8, [sp, #16]
	stur	x8, [x29, #-8]
	b	LBB0_10
LBB0_10:
	ldur	x0, [x29, #-8]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

