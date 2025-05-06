	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #112
	.cfi_def_cfa_offset 112
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	add	x29, sp, #96
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	stur	x1, [x29, #-16]
	stur	x2, [x29, #-24]
	stur	x3, [x29, #-32]
	ldur	x0, [x29, #-8]
	bl	_strlen
	mov	x8, x0
	stur	w8, [x29, #-36]
	ldur	w8, [x29, #-36]
	add	w9, w8, #1
                                        ; implicit-def: $x8
	mov	x8, x9
	sxtw	x8, w8
	lsr	x0, x8, #0
	bl	_malloc
	str	x0, [sp, #48]
	str	wzr, [sp, #44]
	str	wzr, [sp, #40]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #40]
	ldrsb	w8, [x8, x9]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_11
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-16]
	str	x8, [sp, #32]
	strb	wzr, [sp, #31]
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [sp, #32]
	ldrsb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_7
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #40]
	ldrsb	w8, [x8, x9]
	ldr	x9, [sp, #32]
	ldrsb	w9, [x9]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	strb	w8, [sp, #31]
	b	LBB0_7
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #32]
	add	x8, x8, #1
	str	x8, [sp, #32]
	b	LBB0_3
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldrb	w8, [sp, #31]
	tbnz	w8, #0, LBB0_9
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #40]
	add	x8, x8, x9
	ldrb	w8, [x8]
	ldr	x9, [sp, #48]
	ldrsw	x10, [sp, #44]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp, #44]
	add	x9, x9, x10
	strb	w8, [x9]
	b	LBB0_9
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #40]
	add	w8, w8, #1
	str	w8, [sp, #44]
	b	LBB0_1
LBB0_11:
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #44]
	add	x8, x8, x9
	strb	wzr, [x8]
	ldr	x0, [sp, #48]
	bl	_strlen
	mov	x8, x0
	str	w8, [sp, #24]
	mov	w8, #1
	strb	w8, [sp, #23]
	str	wzr, [sp, #16]
	b	LBB0_12
LBB0_12:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #24]
	mov	w10, #2
	sdiv	w9, w9, w10
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_17
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_12 Depth=1
	ldr	x8, [sp, #48]
	ldrsw	x9, [sp, #16]
	ldrsb	w8, [x8, x9]
	ldr	x9, [sp, #48]
	ldr	w10, [sp, #20]
	subs	w10, w10, #1
	ldr	w11, [sp, #16]
	subs	w10, w10, w11
	ldrsb	w9, [x9, w10, sxtw]
	subs	w8, w8, w9
	cset	w8, eq
	tbnz	w8, #0, LBB0_15
	b	LBB0_14
LBB0_14:
	strb	wzr, [sp, #23]
	b	LBB0_17
LBB0_15:                                ;   in Loop: Header=BB0_12 Depth=1
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_12 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_12
LBB0_17:
	ldur	x0, [x29, #-24]
	ldr	x1, [sp, #48]
	mov	x2, #-1
	str	x2, [sp, #8]                    ; 8-byte Folded Spill
	bl	___strcpy_chk
	ldr	x2, [sp, #8]                    ; 8-byte Folded Reload
	ldur	x0, [x29, #-32]
	ldrb	w8, [sp, #23]
	and	w10, w8, #0x1
	adrp	x9, l_.str.1@PAGE
	add	x9, x9, l_.str.1@PAGEOFF
	adrp	x8, l_.str@PAGE
	add	x8, x8, l_.str@PAGEOFF
	ands	w10, w10, #0x1
	csel	x1, x8, x9, ne
	bl	___strcpy_chk
	ldr	x0, [sp, #48]
	bl	_free
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	add	sp, sp, #112
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"True"

l_.str.1:                               ; @.str.1
	.asciz	"False"

.subsections_via_symbols

