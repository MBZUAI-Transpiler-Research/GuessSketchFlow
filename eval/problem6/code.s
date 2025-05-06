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
	stur	w2, [x29, #-16]
	str	x3, [sp, #24]
	ldur	w8, [x29, #-12]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	ldur	w9, [x29, #-12]
	mov	w8, #2
	mul	w8, w8, w9
	subs	w8, w8, #1
	str	w8, [sp, #4]                    ; 4-byte Folded Spill
	b	LBB0_3
LBB0_2:
	mov	w8, #0
	str	w8, [sp, #4]                    ; 4-byte Folded Spill
	b	LBB0_3
LBB0_3:
	ldr	w8, [sp, #4]                    ; 4-byte Folded Reload
	ldr	x9, [sp, #24]
	str	w8, [x9]
	ldr	x8, [sp, #24]
	ldrsw	x9, [x8]
	mov	x8, #4
	mul	x0, x8, x9
	bl	_malloc
	str	x0, [sp, #16]
	ldur	w8, [x29, #-12]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:
	ldur	x8, [x29, #-8]
	ldr	w8, [x8]
	ldr	x9, [sp, #16]
	str	w8, [x9]
	b	LBB0_5
LBB0_5:
	mov	w8, #1
	str	w8, [sp, #12]
	str	w8, [sp, #8]
	b	LBB0_6
LBB0_6:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	ldur	w9, [x29, #-12]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_9
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_6 Depth=1
	ldur	w8, [x29, #-16]
	ldr	x9, [sp, #16]
	ldrsw	x10, [sp, #8]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp, #8]
	str	w8, [x9, x10, lsl #2]
	ldur	x8, [x29, #-8]
	ldrsw	x9, [sp, #12]
	ldr	w8, [x8, x9, lsl #2]
	ldr	x9, [sp, #16]
	ldrsw	x10, [sp, #8]
	mov	x11, x10
	add	w11, w11, #1
	str	w11, [sp, #8]
	str	w8, [x9, x10, lsl #2]
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_6 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_6
LBB0_9:
	ldr	x0, [sp, #16]
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

