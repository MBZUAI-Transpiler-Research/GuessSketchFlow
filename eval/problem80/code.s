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
	stur	w0, [x29, #-12]
	mov	x0, #64
	bl	_malloc
	stur	x0, [x29, #-24]
	ldur	x8, [x29, #-24]
	subs	x8, x8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
                                        ; kill: def $x8 killed $xzr
	stur	xzr, [x29, #-8]
	b	LBB0_11
LBB0_2:
	mov	w8, #62
	stur	w8, [x29, #-28]
	ldur	x8, [x29, #-24]
	strb	wzr, [x8, #63]
	ldur	w8, [x29, #-12]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:
	ldur	x8, [x29, #-24]
	ldursw	x9, [x29, #-28]
	mov	x10, x9
	subs	w10, w10, #1
	stur	w10, [x29, #-28]
	add	x9, x8, x9
	mov	w8, #48
	strb	w8, [x9]
	b	LBB0_8
LBB0_4:
	b	LBB0_5
LBB0_5:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-12]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_5 Depth=1
	ldur	w8, [x29, #-12]
	mov	w9, #2
	sdiv	w10, w8, w9
	mul	w10, w10, w9
	subs	w8, w8, w10
	add	w8, w8, #48
	ldur	x10, [x29, #-24]
	ldursw	x11, [x29, #-28]
	mov	x12, x11
	subs	w12, w12, #1
	stur	w12, [x29, #-28]
	strb	w8, [x10, x11]
	ldur	w8, [x29, #-12]
	sdiv	w8, w8, w9
	stur	w8, [x29, #-12]
	b	LBB0_5
LBB0_7:
	b	LBB0_8
LBB0_8:
	ldur	x8, [x29, #-24]
	ldursw	x9, [x29, #-28]
	mov	x10, x9
	subs	w10, w10, #1
	stur	w10, [x29, #-28]
	add	x9, x8, x9
	mov	w8, #98
	strb	w8, [x9]
	ldur	x8, [x29, #-24]
	ldursw	x9, [x29, #-28]
	mov	x10, x9
	subs	w10, w10, #1
	stur	w10, [x29, #-28]
	add	x9, x8, x9
	mov	w8, #100
	strb	w8, [x9]
	ldur	w8, [x29, #-28]
	add	w8, w8, #1
	str	w8, [sp, #32]
	ldr	w9, [sp, #32]
	mov	w8, #62
	subs	w8, w8, w9
	str	w8, [sp, #28]
	ldr	w8, [sp, #28]
	add	w9, w8, #3
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x0, w8
	bl	_malloc
	str	x0, [sp, #16]
	ldr	x8, [sp, #16]
	subs	x8, x8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	ldur	x0, [x29, #-24]
	bl	_free
                                        ; kill: def $x8 killed $xzr
	stur	xzr, [x29, #-8]
	b	LBB0_11
LBB0_10:
	ldr	x0, [sp, #16]
	ldur	x8, [x29, #-24]
	ldrsw	x9, [sp, #32]
	add	x1, x8, x9
	mov	x2, #-1
	str	x2, [sp, #8]                    ; 8-byte Folded Spill
	bl	___strcpy_chk
	ldr	x2, [sp, #8]                    ; 8-byte Folded Reload
	ldr	x0, [sp, #16]
	adrp	x1, l_.str@PAGE
	add	x1, x1, l_.str@PAGEOFF
	bl	___strcat_chk
	ldur	x0, [x29, #-24]
	bl	_free
	ldr	x8, [sp, #16]
	stur	x8, [x29, #-8]
	b	LBB0_11
LBB0_11:
	ldur	x0, [x29, #-8]
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"db"

.subsections_via_symbols

