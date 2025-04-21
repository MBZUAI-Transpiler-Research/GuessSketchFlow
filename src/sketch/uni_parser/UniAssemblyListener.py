# Generated from UniAssembly.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .UniAssemblyParser import UniAssemblyParser
else:
    from UniAssemblyParser import UniAssemblyParser

# This class defines a complete listener for a parse tree produced by UniAssemblyParser.
class UniAssemblyListener(ParseTreeListener):

    # Enter a parse tree produced by UniAssemblyParser#program.
    def enterProgram(self, ctx:UniAssemblyParser.ProgramContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#program.
    def exitProgram(self, ctx:UniAssemblyParser.ProgramContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#line.
    def enterLine(self, ctx:UniAssemblyParser.LineContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#line.
    def exitLine(self, ctx:UniAssemblyParser.LineContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#instruction.
    def enterInstruction(self, ctx:UniAssemblyParser.InstructionContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#instruction.
    def exitInstruction(self, ctx:UniAssemblyParser.InstructionContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#label.
    def enterLabel(self, ctx:UniAssemblyParser.LabelContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#label.
    def exitLabel(self, ctx:UniAssemblyParser.LabelContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#directive.
    def enterDirective(self, ctx:UniAssemblyParser.DirectiveContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#directive.
    def exitDirective(self, ctx:UniAssemblyParser.DirectiveContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#operands.
    def enterOperands(self, ctx:UniAssemblyParser.OperandsContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#operands.
    def exitOperands(self, ctx:UniAssemblyParser.OperandsContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#segmented_memory_operand.
    def enterSegmented_memory_operand(self, ctx:UniAssemblyParser.Segmented_memory_operandContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#segmented_memory_operand.
    def exitSegmented_memory_operand(self, ctx:UniAssemblyParser.Segmented_memory_operandContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#memory_operand.
    def enterMemory_operand(self, ctx:UniAssemblyParser.Memory_operandContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#memory_operand.
    def exitMemory_operand(self, ctx:UniAssemblyParser.Memory_operandContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#modified_symbol.
    def enterModified_symbol(self, ctx:UniAssemblyParser.Modified_symbolContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#modified_symbol.
    def exitModified_symbol(self, ctx:UniAssemblyParser.Modified_symbolContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#relocation_expr.
    def enterRelocation_expr(self, ctx:UniAssemblyParser.Relocation_exprContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#relocation_expr.
    def exitRelocation_expr(self, ctx:UniAssemblyParser.Relocation_exprContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#expression.
    def enterExpression(self, ctx:UniAssemblyParser.ExpressionContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#expression.
    def exitExpression(self, ctx:UniAssemblyParser.ExpressionContext):
        pass


    # Enter a parse tree produced by UniAssemblyParser#operand.
    def enterOperand(self, ctx:UniAssemblyParser.OperandContext):
        pass

    # Exit a parse tree produced by UniAssemblyParser#operand.
    def exitOperand(self, ctx:UniAssemblyParser.OperandContext):
        pass



del UniAssemblyParser