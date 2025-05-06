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
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	stur	x8, [x29, #-16]
	stur	wzr, [x29, #-20]
	str	wzr, [sp, #24]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #24]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	mov	w9, #0
	str	w9, [sp, #20]                   ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_3
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	lsl	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	str	x8, [sp, #8]                    ; 8-byte Folded Spill
	ldur	x0, [x29, #-8]
	bl	_strlen
	ldr	x8, [sp, #8]                    ; 8-byte Folded Reload
	subs	x8, x8, x0
	cset	w8, lo
	str	w8, [sp, #20]                   ; 4-byte Folded Spill
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]                   ; 4-byte Folded Reload
	tbz	w8, #0, LBB0_8
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x0, [x29, #-16]
	ldur	x8, [x29, #-8]
	ldr	w9, [sp, #24]
	lsl	w9, w9, #1
	ldrsb	w1, [x8, w9, sxtw]
	bl	_strchr
	subs	x8, x0, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-20]
	add	w8, w8, #1
	stur	w8, [x29, #-20]
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	add	w8, w8, #1
	str	w8, [sp, #24]
	b	LBB0_1
LBB0_8:
	ldur	w0, [x29, #-20]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"AEIOU"

.subsections_via_symbols

