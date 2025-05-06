	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	w0, [sp, #28]
	mov	w8, #1
	str	w8, [sp, #24]
	str	wzr, [sp, #20]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #28]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_5
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #28]
	mov	w10, #10
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	str	w8, [sp, #16]
	ldr	w8, [sp, #16]
	mov	w10, #2
	sdiv	w9, w8, w10
	mul	w9, w9, w10
	subs	w8, w8, w9
	subs	w8, w8, #1
	cset	w8, ne
	tbnz	w8, #0, LBB0_4
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	mov	w8, #1
	str	w8, [sp, #20]
	ldr	w9, [sp, #16]
	ldr	w8, [sp, #24]
	mul	w8, w8, w9
	str	w8, [sp, #24]
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #28]
	mov	w9, #10
	sdiv	w8, w8, w9
	str	w8, [sp, #28]
	b	LBB0_1
LBB0_5:
	ldr	w8, [sp, #20]
	subs	w8, w8, #0
	cset	w8, eq
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:
	ldr	w8, [sp, #24]
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	b	LBB0_8
LBB0_7:
	mov	w8, #0
	str	w8, [sp, #11]                   ; 4-byte Folded Spill
	b	LBB0_8
LBB0_8:
	ldr	w0, [sp, #11]                   ; 4-byte Folded Reload
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

