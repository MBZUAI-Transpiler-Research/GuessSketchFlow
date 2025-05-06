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
	stur	w0, [x29, #-4]
	stur	x1, [x29, #-16]
	str	x2, [sp, #24]
	mov	w8, #10
	str	w8, [sp, #20]
	ldr	x9, [sp, #24]
	mov	w8, #1
	str	w8, [x9]
	ldur	x9, [x29, #-16]
	str	w8, [x9]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w8, [x29, #-4]
	subs	w8, w8, #1
	cset	w8, eq
	tbnz	w8, #0, LBB0_8
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-4]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #24]
	ldr	w8, [x8]
	ldr	w9, [sp, #20]
	subs	w8, w8, w9
	cset	w8, lt
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #20]
	lsl	w8, w8, #1
	str	w8, [sp, #20]
	ldur	x0, [x29, #-16]
	ldrsw	x8, [sp, #20]
	lsl	x1, x8, #2
	bl	_realloc
	stur	x0, [x29, #-16]
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-4]
	ldur	x9, [x29, #-16]
	ldr	x12, [sp, #24]
	ldrsw	x10, [x12]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [x12]
	str	w8, [x9, x10, lsl #2]
	ldur	w8, [x29, #-4]
	mov	w9, #3
	mul	w8, w8, w9
	add	w8, w8, #1
	stur	w8, [x29, #-4]
	b	LBB0_7
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldur	w8, [x29, #-4]
	mov	w9, #2
	sdiv	w8, w8, w9
	stur	w8, [x29, #-4]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_1
LBB0_8:
	mov	w8, #1
	stur	w8, [x29, #-2]
	b	LBB0_9
LBB0_9:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_11 Depth 2
	ldr	w8, [sp, #16]
	ldr	x9, [sp, #24]
	ldr	w9, [x9]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_17
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_9 Depth=1
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #12]
	ldr	w8, [sp, #16]
	subs	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_11
LBB0_11:                                ;   Parent Loop BB0_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #8]
	subs	w8, w8, #0
	cset	w8, lt
	mov	w9, #0
	str	w9, [sp, #4]                    ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_13
	b	LBB0_12
LBB0_12:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #8]
	ldr	w8, [x8, x9, lsl #2]
	ldr	w9, [sp, #12]
	subs	w8, w8, w9
	cset	w8, gt
	str	w8, [sp, #4]                    ; 4-byte Folded Spill
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_11 Depth=2
	ldr	w8, [sp, #4]                    ; 4-byte Folded Reload
	tbz	w8, #0, LBB0_15
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_11 Depth=2
	ldur	x8, [x29, #-16]
	ldrsw	x9, [sp, #8]
	ldr	w8, [x8, x9, lsl #2]
	ldur	x9, [x29, #-16]
	ldr	w10, [sp, #8]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	ldr	w8, [sp, #8]
	subs	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_11
LBB0_15:                                ;   in Loop: Header=BB0_9 Depth=1
	ldr	w8, [sp, #12]
	ldur	x9, [x29, #-16]
	ldr	w10, [sp, #8]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_9 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_9
LBB0_17:
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

