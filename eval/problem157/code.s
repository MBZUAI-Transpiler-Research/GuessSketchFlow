	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_func0                          ; -- Begin function func0
	.p2align	2
_func0:                                 ; @func0
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #220
	.cfi_def_cfa_offset 220
	stp	x29, x30, [sp, #208]            ; 16-byte Folded Spill
	add	x29, sp, #208
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	stur	x8, [x29, #-8]
	str	w0, [sp, #28]
	str	x1, [sp, #16]
	add	x0, sp, #96
	adrp	x1, l___const.func0.rep@PAGE
	add	x1, x1, l___const.func0.rep@PAGEOFF
	mov	x2, #104
	bl	_memcpy
	add	x0, sp, #32
	adrp	x1, l___const.func0.num@PAGE
	add	x1, x1, l___const.func0.num@PAGEOFF
	mov	x2, #52
	bl	_memcpy
	str	wzr, [sp, #12]
	ldr	x8, [sp, #16]
	strb	wzr, [x8]
	b	LBB0_1
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #28]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_8
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_3
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #28]
	ldrsw	x10, [sp, #12]
	add	x9, sp, #32
	ldr	w9, [x9, x10, lsl #2]
	subs	w8, w8, w9
	cset	w8, lt
	tbnz	w8, #0, LBB0_5
	b	LBB0_4
LBB0_4:                                 ;   in Loop: Header=BB0_3 Depth=2
	ldr	x0, [sp, #16]
	ldrsw	x9, [sp, #12]
	add	x8, sp, #96
	ldr	x1, [x8, x9, lsl #3]
	mov	x2, #-1
	bl	___strcat_chk
	ldrsw	x9, [sp, #12]
	add	x8, sp, #33
	ldr	w9, [x8, x9, lsl #2]
	ldr	w8, [sp, #28]
	subs	w8, w8, w9
	str	w8, [sp, #36]
	b	LBB0_3
LBB0_5:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #36]
	subs	w8, w8, #0
	cset	w8, le
	tbnz	w8, #0, LBB0_7
	b	LBB0_6
LBB0_6:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_7
LBB0_7:                                 ;   in Loop: Header=BB0_1 Depth=1
	b	LBB0_1
LBB0_8:
	ldur	x9, [x29, #-8]
	adrp	x8, ___stack_chk_guard@GOTPAGE
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
	ldr	x8, [x8]
	subs	x8, x8, x9
	cset	w8, eq
	tbnz	w8, #0, LBB0_10
	b	LBB0_9
LBB0_9:
	bl	___stack_chk_fail
LBB0_10:
	ldp	x29, x30, [sp, #208]            ; 16-byte Folded Reload
	add	sp, sp, #220
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"m"

l_.str.1:                               ; @.str.1
	.asciz	"cm"

l_.str.2:                               ; @.str.2
	.asciz	"d"

l_.str.3:                               ; @.str.3
	.asciz	"cd"

l_.str.4:                               ; @.str.4
	.asciz	"c"

l_.str.5:                               ; @.str.5
	.asciz	"xc"

l_.str.6:                               ; @.str.6
	.asciz	"l"

l_.str.7:                               ; @.str.7
	.asciz	"xl"

l_.str.8:                               ; @.str.8
	.asciz	"x"

l_.str.9:                               ; @.str.9
	.asciz	"ix"

l_.str.10:                              ; @.str.10
	.asciz	"v"

l_.str.11:                              ; @.str.11
	.asciz	"iv"

l_.str.12:                              ; @.str.12
	.asciz	"i"

	.section	__DATA,__const
	.p2align	3                               ; @__const.func0.rep
l___const.func0.rep:
	.quad	l_.str
	.quad	l_.str.1
	.quad	l_.str.2
	.quad	l_.str.3
	.quad	l_.str.4
	.quad	l_.str.5
	.quad	l_.str.6
	.quad	l_.str.7
	.quad	l_.str.8
	.quad	l_.str.9
	.quad	l_.str.10
	.quad	l_.str.11
	.quad	l_.str.12

	.section	__TEXT,__const
	.p2align	2                               ; @__const.func0.num
l___const.func0.num:
	.long	1000                            ; 0x3e8
	.long	900                             ; 0x384
	.long	500                             ; 0x1f4
	.long	400                             ; 0x190
	.long	100                             ; 0x64
	.long	90                              ; 0x5a
	.long	50                              ; 0x32
	.long	40                              ; 0x28
	.long	10                              ; 0xa
	.long	9                               ; 0x9
	.long	5                               ; 0x5
	.long	4                               ; 0x4
	.long	1                               ; 0x1

.subsections_via_symbols

