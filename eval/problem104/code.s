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
	stur	w0, [x29, #-12]
	stur	w1, [x29, #-16]
	ldur	w8, [x29, #-12]
	ldur	w9, [x29, #-16]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	stur	x8, [x29, #-8]
	b	LBB0_6
LBB0_2:
	ldur	w8, [x29, #-16]
	ldur	w9, [x29, #-12]
	add	w8, w8, w9
	mov	w9, #2
	str	w9, [sp, #8]                    ; 4-byte Folded Spill
	sdiv	w8, w8, w9
	stur	w8, [x29, #12]
	mov	x0, #33
	bl	_malloc
	ldr	w8, [sp, #8]                    ; 4-byte Folded Reload
	str	x0, [sp, #16]
	ldr	x9, [sp, #16]
	strb	wzr, [x9]
	str	w8, [sp, #12]
	ldr	x8, [sp, #16]
	ldrsw	x9, [sp, #11]
	mov	x10, x9
	subs	w10, w10, #1
	str	w10, [sp, #11]
	add	x8, x8, x9
	strb	wzr, [x8]
	b	LBB0_3
LBB0_3:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-20]
	mov	w9, #2
	sdiv	w10, w8, w9
	mul	w10, w10, w9
	subs	w8, w8, w10
	add	w8, w8, #48
	ldr	x10, [sp, #16]
	ldrsw	x11, [sp, #11]
	mov	x12, x11
	subs	w12, w12, #1
	str	w12, [sp, #11]
	strb	w8, [x10, x11]
	ldur	w8, [x29, #-20]
	sdiv	w8, w8, w9
	stur	w8, [x29, #-20]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=1
	ldur	w8, [x29, #-20]
	subs	w8, w8, #0
	cset	w8, gt
	tbnz	w8, #0, LBB0_3
	b	LBB0_5
LBB0_5:
	ldr	x8, [sp, #16]
	ldr	w9, [sp, #11]
	add	w9, w9, #1
	add	x8, x8, w9, sxtw
	stur	x8, [x29, #-8]
	b	LBB0_6
LBB0_6:
	ldur	x0, [x29, #-8]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"-1"

.subsections_via_symbols

