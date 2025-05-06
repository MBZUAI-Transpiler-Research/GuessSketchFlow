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
	str	wzr, [sp, #32]
	str	wzr, [sp, #28]
	str	wzr, [sp, #16]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_4 Depth 2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #36]
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_17
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp, #40]
	ldrsw	x9, [sp, #16]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #24]
	ldr	w8, [sp, #24]
	subs	w8, w8, #1
	cset	w8, le
	tbnz	w8, #0, LBB0_15
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #12]
	mov	w8, #2
	str	w8, [sp, #8]
	b	LBB0_4
LBB0_4:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #8]
	ldr	w9, [sp, #8]
	mul	w8, w8, w9
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, gt
	tbnz	w8, #0, LBB0_9
	b	LBB0_5
LBB0_5:                                 ;   in Loop: Header=BB0_4 Depth=2
	ldr	w8, [sp, #24]
	ldr	w10, [sp, #8]
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #0
	cset	w8, ne
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #12]
	b	LBB0_9
LBB0_7:                                 ;   in Loop: Header=BB0_4 Depth=2
	b	LBB0_8
LBB0_8:                                 ;   in Loop: Header=BB0_4 Depth=2
	ldr	w8, [sp, #8]
	add	w8, w8, #1
	str	w8, [sp, #8]
	b	LBB0_4
LBB0_9:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_14
	b	LBB0_10
LBB0_10:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	ldr	w9, [sp, #31]
	subs	w8, w8, w9
	cset	w8, le
	tbnz	w8, #0, LBB0_12
	b	LBB0_11
LBB0_11:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #24]
	str	w8, [sp, #4]                    ; 4-byte Folded Spill
	b	LBB0_13
LBB0_12:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #31]
	str	w8, [sp, #4]                    ; 4-byte Folded Spill
	b	LBB0_13
LBB0_13:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #4]                    ; 4-byte Folded Reload
	str	w8, [sp, #31]
	b	LBB0_14
LBB0_14:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_15
LBB0_15:                                ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_16
LBB0_16:                                ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	LBB0_1
LBB0_17:
	b	LBB0_18
LBB0_18:                                ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #31]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_20
	b	LBB0_19
LBB0_19:                                ;   in Loop: Header=BB0_18 Depth=1
	ldr	w8, [sp, #31]
	mov	w9, #10
	sdiv	w10, w8, w9
	mul	w10, w10, w9
	subs	w10, w8, w10
	ldr	w8, [sp, #28]
	add	w8, w8, w10
	str	w8, [sp, #28]
	ldr	w8, [sp, #31]
	sdiv	w8, w8, w9
	str	w8, [sp, #31]
	b	LBB0_18
LBB0_20:
	ldr	w0, [sp, #28]
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

