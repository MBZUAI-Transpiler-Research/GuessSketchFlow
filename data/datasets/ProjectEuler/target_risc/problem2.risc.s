	.file	"problem2.c"
	.option pic
	.text
	.section	.rodata
	.align	3
.LC0:
	.string	"%u\n"
	.text
	.align	1
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-32	
	sd	ra,24(sp)	
	sd	s0,16(sp)	
	addi	s0,sp,32	
	li	a5,1		
	sw	a5,-32(s0)	
	li	a5,1		
	sw	a5,-28(s0)	
	li	a5,2		
	sw	a5,-24(s0)	
	sw	zero,-20(s0)	
	j	.L2		
.L3:
	lw	a5,-32(s0)		
	mv	a4,a5	
	lw	a5,-28(s0)		
	addw	a5,a4,a5	
	sw	a5,-24(s0)	
	lw	a5,-24(s0)		
	andi	a5,a5,1	
	sext.w	a5,a5	
	seqz	a5,a5	
	andi	a5,a5,0xff	
	sext.w	a5,a5	
	lw	a4,-24(s0)		
	mulw	a5,a4,a5	
	sext.w	a5,a5	
	lw	a4,-20(s0)		
	addw	a5,a4,a5	
	sw	a5,-20(s0)	
	lw	a5,-28(s0)		
	sw	a5,-32(s0)	
	lw	a5,-24(s0)		
	sw	a5,-28(s0)	
.L2:
	lw	a5,-24(s0)		
	sext.w	a4,a5	
	li	a5,4001792		
	addi	a5,a5,-1793	
	bleu	a4,a5,.L3	
	lw	a5,-20(s0)		
	mv	a1,a5	
	lla	a0,.LC0	
	call	printf@plt	
	li	a5,0		
	mv	a0,a5	
	ld	ra,24(sp)		
	ld	s0,16(sp)		
	addi	sp,sp,32	
	jr	ra		
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0"
	.section	.note.GNU-stack,"",@progbits
