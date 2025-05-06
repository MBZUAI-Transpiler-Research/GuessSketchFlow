	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, #40]
	str	w1, [sp, #36]
	ldr	x8, [sp, #40]
	ldr	x8, [x8]
	str	x8, [sp, #24]
	ldr	x8, [sp, #40]
	ldr	x8, [x8]
	str	x8, [sp, #16]
	mov	w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #36]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_9
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #24]
	subs	x8, x8, #0
	cset	w8, ge
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #24]
	ldr	x9, [sp, #40]
	ldrsw	x10, [sp, #12]
	ldr	x9, [x9, x10, lsl #3]
	add	x8, x8, x9
	str	x8, [sp]                        ; 8-byte Folded Spill
	b	LBB0_5
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #12]
	ldr	x8, [x8, x9, lsl #3]
	str	x8, [sp]                        ; 8-byte Folded Spill
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp]                        ; 8-byte Folded Reload
	str	x8, [sp, #24]
	ldr	x8, [sp, #24]
	ldr	x9, [sp, #16]
	subs	x8, x8, x9
	cset	w8, ge
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #24]
	str	x8, [sp, #16]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_9:
	ldr	x0, [sp, #16]
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

