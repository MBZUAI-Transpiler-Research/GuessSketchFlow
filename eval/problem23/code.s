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
	stur	x0, [x29, #-8]
	stur	x1, [x29, #-16]
	stur	wzr, [x29, #-20]
	ldur	x8, [x29, #-8]
	str	x8, [sp, #32]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
                                        ;     Child Loop BB0_15 Depth 2
	ldr	x8, [sp, #33]
	ldrb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_21
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [sp, #33]
	ldrsb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, eq
	mov	w9, #0
	str	w9, [sp, #16]                   ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_6
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #33]
	ldrsb	w0, [x8]
	bl	_isdigit
	subs	w8, w0, #0
	cset	w8, ne
	mov	w9, #0
	str	w9, [sp, #16]                   ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #33]
	ldrsb	w8, [x8]
	subs	w8, w8, #45
	cset	w8, ne
	str	w8, [sp, #16]                   ; 4-byte Folded Spill
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #16]                   ; 4-byte Folded Reload
	tbz	w8, #0, LBB0_8
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #33]
	add	x8, x8, #1
	str	x8, [sp, #33]
	b	LBB0_3
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #33]
	ldrb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	b	LBB0_21
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x0, [sp, #33]
	add	x1, sp, #24
	mov	w2, #10
	bl	_strtol
	mov	x8, x0
	str	w8, [sp, #20]
	ldr	x8, [sp, #33]
	ldr	x9, [sp, #24]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_14
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #24]
	ldrsb	w8, [x8]
	subs	w8, w8, #44
	cset	w8, eq
	tbnz	w8, #0, LBB0_13
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #24]
	ldrsb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_14
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #22]
	ldursw	x10, [x29, #-20]
	mov	x9, x10
	add	w9, w9, #1
	stur	w9, [x29, #-22]
	adrp	x9, _func0.out@PAGE
	add	x9, x9, _func0.out@PAGEOFF
	str	w8, [x9, x10, lsl #2]
	b	LBB0_20
LBB0_14:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_15
LBB0_15:                                ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [sp, #24]
	ldrsb	w8, [x8]
	subs	w8, w8, #0
	cset	w8, eq
	mov	w9, #0
	str	w9, [sp, #12]                   ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_17
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_15 Depth=2
	ldr	x8, [sp, #24]
	ldrsb	w8, [x8]
	subs	w8, w8, #44
	cset	w8, ne
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	b	LBB0_17
LBB0_17:                                ;   in Loop: Header=BB0_15 Depth=2
	ldr	w8, [sp, #12]                   ; 4-byte Folded Reload
	tbz	w8, #0, LBB0_19
	b	LBB0_18
LBB0_18:                                ;   in Loop: Header=BB0_15 Depth=2
	ldr	x8, [sp, #24]
	add	x8, x8, #1
	str	x8, [sp, #24]
	b	LBB0_15
LBB0_19:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_20
LBB0_20:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #24]
	str	x8, [sp, #33]
	b	LBB0_1
LBB0_21:
	ldur	w8, [x29, #-22]
	ldur	x9, [x29, #-16]
	str	w8, [x9]
	adrp	x0, _func0.out@PAGE
	add	x0, x0, _func0.out@PAGEOFF
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #80
	ret
	.cfi_endproc
                                        ; -- End function
.zerofill __DATA,__bss,_func0.out,1024,2 ; @func0.out
.subsections_via_symbols

