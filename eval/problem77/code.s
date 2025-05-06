	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	w0, [sp, #24]
	str	w1, [sp, #20]
	mov	w8, #1
	str	w8, [sp, #16]
	str	wzr, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, gt
	mov	w9, #0
	str	w9, [sp, #8]                    ; 4-byte Folded Spill
	tbnz	w8, #0, LBB0_3
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	subs	w8, w8, #100
	cset	w8, lt
	str	w8, [sp, #8]                    ; 4-byte Folded Spill
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #8]                    ; 4-byte Folded Reload
	tbz	w8, #0, LBB0_7
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	cset	w8, ne
	tbnz	w8, #0, LBB0_6
	b	LBB0_5
LBB0_5:
	mov	w8, #1
	str	w8, [sp, #28]
	b	LBB0_8
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #20]
	mul	w8, w8, w9
	str	w8, [sp, #16]
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #11]
	b	LBB0_1
LBB0_7:
	str	wzr, [sp, #28]
	b	LBB0_8
LBB0_8:
	ldr	w0, [sp, #28]
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols

