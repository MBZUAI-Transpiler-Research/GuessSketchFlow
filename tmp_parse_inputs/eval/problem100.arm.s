	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	x0, [x29, #-8]
	ldur	x0, [x29, #-8]
	bl	_atof
	str	d0, [sp, #16]
	ldr	d0, [sp, #16]
	fcmp	d0, #0.0
	cset	w8, pl
	tbnz	w8, #0, LBB0_2
	b	LBB0_1
LBB0_1:
	ldr	d0, [sp, #16]
	fmov	d1, #0.50000000
	fsub	d0, d0, d1
	frintp	d0, d0
	str	d0, [sp, #8]                    ; 8-byte Folded Spill
	b	LBB0_3
LBB0_2:
	ldr	d0, [sp, #16]
	fmov	d1, #0.50000000
	fadd	d0, d0, d1
	frintm	d0, d0
	str	d0, [sp, #8]                    ; 8-byte Folded Spill
	b	LBB0_3
LBB0_3:
	ldr	d0, [sp, #8]                    ; 8-byte Folded Reload
	fcvtzs	w0, d0
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
