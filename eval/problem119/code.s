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
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	str	x8, [sp, #24]
	ldur	x0, [x29, #-16]
	bl	_strlen
	str	x0, [sp, #16]
	ldr	x8, [sp, #16]
	subs	x8, x8, #2
                                        ; kill: def $w8 killed $w8 killed $x8
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #11]
	subs	w8, w8, #1
	cset	w8, lt
	tbnz	w8, #0, LBB0_8
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x0, [sp, #24]
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #11]
	ldrsb	w1, [x8, x9]
	bl	_strchr
	subs	x8, x0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_6
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x0, [sp, #24]
	ldur	x8, [x29, #-16]
	ldr	w9, [sp, #11]
	add	w9, w9, #1
	ldrsb	w1, [x8, w9, sxtw]
	bl	_strchr
	subs	x8, x0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x0, [sp, #24]
	ldur	x8, [x29, #-16]
	ldr	w9, [sp, #11]
	subs	w9, w9, #1
	ldrsb	w1, [x8, w9, sxtw]
	bl	_strchr
	subs	x8, x0, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #11]
	add	x9, x8, x9
	ldrb	w9, [x9]
	adrp	x10, _func0.out@PAGE
	adrp	x8, _func0.out@PAGE
	add	x8, x8, _func0.out@PAGEOFF
	strb	w9, [x10, _func0.out@PAGEOFF]
	stur	x8, [x29, #-8]
	b	LBB0_9
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #11]
	subs	w8, w8, #1
	str	w8, [sp, #11]
	b	LBB0_1
LBB0_8:
	adrp	x9, _func0.out@PAGE
	adrp	x8, _func0.out@PAGE
	add	x8, x8, _func0.out@PAGEOFF
	strb	wzr, [x9, _func0.out@PAGEOFF]
	stur	x8, [x29, #-8]
	b	LBB0_9
LBB0_9:
	ldur	x0, [x29, #-8]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.zerofill __DATA,__bss,_func0.out,2,0   ; @func0.out
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"AEIOUaeiou"

.subsections_via_symbols

