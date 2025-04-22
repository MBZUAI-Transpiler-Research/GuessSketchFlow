# Generated from UniAssembly.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,21,149,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,1,0,1,0,1,0,5,0,28,
        8,0,10,0,12,0,31,9,0,1,0,1,0,1,1,1,1,1,1,3,1,38,8,1,1,2,1,2,3,2,
        42,8,2,1,3,1,3,1,3,5,3,47,8,3,10,3,12,3,50,9,3,1,3,1,3,1,3,1,3,3,
        3,56,8,3,1,4,1,4,1,4,5,4,61,8,4,10,4,12,4,64,9,4,1,4,1,4,3,4,68,
        8,4,1,5,1,5,1,5,5,5,73,8,5,10,5,12,5,76,9,5,1,6,1,6,1,6,1,6,1,7,
        1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,
        1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,3,7,
        114,8,7,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,3,8,124,8,8,1,9,1,9,1,9,
        1,9,1,9,1,9,1,10,1,10,1,10,1,10,1,11,1,11,1,11,1,11,1,11,1,11,1,
        11,1,11,1,11,1,11,1,11,3,11,147,8,11,1,11,0,0,12,0,2,4,6,8,10,12,
        14,16,18,20,22,0,0,162,0,24,1,0,0,0,2,37,1,0,0,0,4,39,1,0,0,0,6,
        55,1,0,0,0,8,62,1,0,0,0,10,69,1,0,0,0,12,77,1,0,0,0,14,113,1,0,0,
        0,16,123,1,0,0,0,18,125,1,0,0,0,20,131,1,0,0,0,22,146,1,0,0,0,24,
        29,3,2,1,0,25,26,5,20,0,0,26,28,3,2,1,0,27,25,1,0,0,0,28,31,1,0,
        0,0,29,27,1,0,0,0,29,30,1,0,0,0,30,32,1,0,0,0,31,29,1,0,0,0,32,33,
        5,0,0,1,33,1,1,0,0,0,34,38,3,4,2,0,35,38,3,6,3,0,36,38,3,8,4,0,37,
        34,1,0,0,0,37,35,1,0,0,0,37,36,1,0,0,0,38,3,1,0,0,0,39,41,5,2,0,
        0,40,42,3,10,5,0,41,40,1,0,0,0,41,42,1,0,0,0,42,5,1,0,0,0,43,44,
        5,18,0,0,44,45,5,2,0,0,45,47,5,19,0,0,46,43,1,0,0,0,47,50,1,0,0,
        0,48,46,1,0,0,0,48,49,1,0,0,0,49,51,1,0,0,0,50,48,1,0,0,0,51,52,
        5,3,0,0,52,56,5,8,0,0,53,54,5,2,0,0,54,56,5,8,0,0,55,48,1,0,0,0,
        55,53,1,0,0,0,56,7,1,0,0,0,57,58,5,18,0,0,58,59,5,2,0,0,59,61,5,
        19,0,0,60,57,1,0,0,0,61,64,1,0,0,0,62,60,1,0,0,0,62,63,1,0,0,0,63,
        65,1,0,0,0,64,62,1,0,0,0,65,67,5,3,0,0,66,68,3,10,5,0,67,66,1,0,
        0,0,67,68,1,0,0,0,68,9,1,0,0,0,69,74,3,22,11,0,70,71,5,9,0,0,71,
        73,3,22,11,0,72,70,1,0,0,0,73,76,1,0,0,0,74,72,1,0,0,0,74,75,1,0,
        0,0,75,11,1,0,0,0,76,74,1,0,0,0,77,78,5,1,0,0,78,79,5,8,0,0,79,80,
        5,6,0,0,80,13,1,0,0,0,81,82,5,6,0,0,82,83,5,10,0,0,83,84,5,1,0,0,
        84,114,5,11,0,0,85,86,5,10,0,0,86,87,5,1,0,0,87,114,5,11,0,0,88,
        89,5,12,0,0,89,90,5,1,0,0,90,91,5,9,0,0,91,92,3,18,9,0,92,93,5,13,
        0,0,93,114,1,0,0,0,94,95,5,12,0,0,95,96,5,1,0,0,96,114,5,13,0,0,
        97,98,5,12,0,0,98,99,5,1,0,0,99,100,5,9,0,0,100,101,5,6,0,0,101,
        114,5,13,0,0,102,103,5,12,0,0,103,104,5,1,0,0,104,105,5,9,0,0,105,
        106,5,4,0,0,106,114,5,13,0,0,107,108,5,12,0,0,108,109,5,1,0,0,109,
        110,5,9,0,0,110,111,5,6,0,0,111,112,5,13,0,0,112,114,5,15,0,0,113,
        81,1,0,0,0,113,85,1,0,0,0,113,88,1,0,0,0,113,94,1,0,0,0,113,97,1,
        0,0,0,113,102,1,0,0,0,113,107,1,0,0,0,114,15,1,0,0,0,115,116,5,8,
        0,0,116,117,5,2,0,0,117,118,5,8,0,0,118,124,5,2,0,0,119,120,5,8,
        0,0,120,121,5,2,0,0,121,122,5,8,0,0,122,124,5,3,0,0,123,115,1,0,
        0,0,123,119,1,0,0,0,124,17,1,0,0,0,125,126,5,14,0,0,126,127,5,8,
        0,0,127,128,5,2,0,0,128,129,5,8,0,0,129,130,5,2,0,0,130,19,1,0,0,
        0,131,132,5,16,0,0,132,133,5,17,0,0,133,134,5,2,0,0,134,21,1,0,0,
        0,135,147,5,1,0,0,136,147,3,12,6,0,137,147,3,14,7,0,138,147,3,16,
        8,0,139,147,3,20,10,0,140,147,5,6,0,0,141,147,5,4,0,0,142,147,5,
        5,0,0,143,147,5,2,0,0,144,147,5,3,0,0,145,147,5,7,0,0,146,135,1,
        0,0,0,146,136,1,0,0,0,146,137,1,0,0,0,146,138,1,0,0,0,146,139,1,
        0,0,0,146,140,1,0,0,0,146,141,1,0,0,0,146,142,1,0,0,0,146,143,1,
        0,0,0,146,144,1,0,0,0,146,145,1,0,0,0,147,23,1,0,0,0,11,29,37,41,
        48,55,62,67,74,113,123,146
    ]

class UniAssemblyParser ( Parser ):

    grammarFileName = "UniAssembly.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "':'", "','", "'('", "')'", "'['", "']'", "'#'", "'!'", 
                     "'.'", "'-'", "'{'", "'}'" ]

    symbolicNames = [ "<INVALID>", "REGISTER", "SYMBOL", "DOT_SYMBOL", "IMMEDIATE", 
                      "BINARY_IMMEDIATE", "NUMBER", "STRING", "COLON", "COMMA", 
                      "LPAREN", "RPAREN", "LBRACK", "RBRACK", "HASH", "BANG", 
                      "DOT", "MINUS", "LCURLY", "RCURLY", "NEWLINE", "WS" ]

    RULE_program = 0
    RULE_line = 1
    RULE_instruction = 2
    RULE_label = 3
    RULE_directive = 4
    RULE_operands = 5
    RULE_segmented_memory_operand = 6
    RULE_memory_operand = 7
    RULE_modified_symbol = 8
    RULE_relocation_expr = 9
    RULE_expression = 10
    RULE_operand = 11

    ruleNames =  [ "program", "line", "instruction", "label", "directive", 
                   "operands", "segmented_memory_operand", "memory_operand", 
                   "modified_symbol", "relocation_expr", "expression", "operand" ]

    EOF = Token.EOF
    REGISTER=1
    SYMBOL=2
    DOT_SYMBOL=3
    IMMEDIATE=4
    BINARY_IMMEDIATE=5
    NUMBER=6
    STRING=7
    COLON=8
    COMMA=9
    LPAREN=10
    RPAREN=11
    LBRACK=12
    RBRACK=13
    HASH=14
    BANG=15
    DOT=16
    MINUS=17
    LCURLY=18
    RCURLY=19
    NEWLINE=20
    WS=21

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def line(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(UniAssemblyParser.LineContext)
            else:
                return self.getTypedRuleContext(UniAssemblyParser.LineContext,i)


        def EOF(self):
            return self.getToken(UniAssemblyParser.EOF, 0)

        def NEWLINE(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.NEWLINE)
            else:
                return self.getToken(UniAssemblyParser.NEWLINE, i)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_program

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgram" ):
                listener.enterProgram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgram" ):
                listener.exitProgram(self)




    def program(self):

        localctx = UniAssemblyParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 24
            self.line()
            self.state = 29
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==20:
                self.state = 25
                self.match(UniAssemblyParser.NEWLINE)
                self.state = 26
                self.line()
                self.state = 31
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 32
            self.match(UniAssemblyParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LineContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def instruction(self):
            return self.getTypedRuleContext(UniAssemblyParser.InstructionContext,0)


        def label(self):
            return self.getTypedRuleContext(UniAssemblyParser.LabelContext,0)


        def directive(self):
            return self.getTypedRuleContext(UniAssemblyParser.DirectiveContext,0)


        def getRuleIndex(self):
            return UniAssemblyParser.RULE_line

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLine" ):
                listener.enterLine(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLine" ):
                listener.exitLine(self)




    def line(self):

        localctx = UniAssemblyParser.LineContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_line)
        try:
            self.state = 37
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 34
                self.instruction()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 35
                self.label()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 36
                self.directive()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InstructionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SYMBOL(self):
            return self.getToken(UniAssemblyParser.SYMBOL, 0)

        def operands(self):
            return self.getTypedRuleContext(UniAssemblyParser.OperandsContext,0)


        def getRuleIndex(self):
            return UniAssemblyParser.RULE_instruction

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInstruction" ):
                listener.enterInstruction(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInstruction" ):
                listener.exitInstruction(self)




    def instruction(self):

        localctx = UniAssemblyParser.InstructionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_instruction)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 39
            self.match(UniAssemblyParser.SYMBOL)
            self.state = 41
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 71166) != 0):
                self.state = 40
                self.operands()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LabelContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DOT_SYMBOL(self):
            return self.getToken(UniAssemblyParser.DOT_SYMBOL, 0)

        def COLON(self):
            return self.getToken(UniAssemblyParser.COLON, 0)

        def LCURLY(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.LCURLY)
            else:
                return self.getToken(UniAssemblyParser.LCURLY, i)

        def SYMBOL(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.SYMBOL)
            else:
                return self.getToken(UniAssemblyParser.SYMBOL, i)

        def RCURLY(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.RCURLY)
            else:
                return self.getToken(UniAssemblyParser.RCURLY, i)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_label

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLabel" ):
                listener.enterLabel(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLabel" ):
                listener.exitLabel(self)




    def label(self):

        localctx = UniAssemblyParser.LabelContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_label)
        self._la = 0 # Token type
        try:
            self.state = 55
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [3, 18]:
                self.enterOuterAlt(localctx, 1)
                self.state = 48
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==18:
                    self.state = 43
                    self.match(UniAssemblyParser.LCURLY)
                    self.state = 44
                    self.match(UniAssemblyParser.SYMBOL)
                    self.state = 45
                    self.match(UniAssemblyParser.RCURLY)
                    self.state = 50
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 51
                self.match(UniAssemblyParser.DOT_SYMBOL)
                self.state = 52
                self.match(UniAssemblyParser.COLON)
                pass
            elif token in [2]:
                self.enterOuterAlt(localctx, 2)
                self.state = 53
                self.match(UniAssemblyParser.SYMBOL)
                self.state = 54
                self.match(UniAssemblyParser.COLON)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DirectiveContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DOT_SYMBOL(self):
            return self.getToken(UniAssemblyParser.DOT_SYMBOL, 0)

        def LCURLY(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.LCURLY)
            else:
                return self.getToken(UniAssemblyParser.LCURLY, i)

        def SYMBOL(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.SYMBOL)
            else:
                return self.getToken(UniAssemblyParser.SYMBOL, i)

        def RCURLY(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.RCURLY)
            else:
                return self.getToken(UniAssemblyParser.RCURLY, i)

        def operands(self):
            return self.getTypedRuleContext(UniAssemblyParser.OperandsContext,0)


        def getRuleIndex(self):
            return UniAssemblyParser.RULE_directive

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDirective" ):
                listener.enterDirective(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDirective" ):
                listener.exitDirective(self)




    def directive(self):

        localctx = UniAssemblyParser.DirectiveContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_directive)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 62
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==18:
                self.state = 57
                self.match(UniAssemblyParser.LCURLY)
                self.state = 58
                self.match(UniAssemblyParser.SYMBOL)
                self.state = 59
                self.match(UniAssemblyParser.RCURLY)
                self.state = 64
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 65
            self.match(UniAssemblyParser.DOT_SYMBOL)
            self.state = 67
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 71166) != 0):
                self.state = 66
                self.operands()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class OperandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def operand(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(UniAssemblyParser.OperandContext)
            else:
                return self.getTypedRuleContext(UniAssemblyParser.OperandContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.COMMA)
            else:
                return self.getToken(UniAssemblyParser.COMMA, i)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_operands

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOperands" ):
                listener.enterOperands(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOperands" ):
                listener.exitOperands(self)




    def operands(self):

        localctx = UniAssemblyParser.OperandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_operands)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 69
            self.operand()
            self.state = 74
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==9:
                self.state = 70
                self.match(UniAssemblyParser.COMMA)
                self.state = 71
                self.operand()
                self.state = 76
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Segmented_memory_operandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REGISTER(self):
            return self.getToken(UniAssemblyParser.REGISTER, 0)

        def COLON(self):
            return self.getToken(UniAssemblyParser.COLON, 0)

        def NUMBER(self):
            return self.getToken(UniAssemblyParser.NUMBER, 0)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_segmented_memory_operand

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSegmented_memory_operand" ):
                listener.enterSegmented_memory_operand(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSegmented_memory_operand" ):
                listener.exitSegmented_memory_operand(self)




    def segmented_memory_operand(self):

        localctx = UniAssemblyParser.Segmented_memory_operandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_segmented_memory_operand)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 77
            self.match(UniAssemblyParser.REGISTER)
            self.state = 78
            self.match(UniAssemblyParser.COLON)
            self.state = 79
            self.match(UniAssemblyParser.NUMBER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Memory_operandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUMBER(self):
            return self.getToken(UniAssemblyParser.NUMBER, 0)

        def LPAREN(self):
            return self.getToken(UniAssemblyParser.LPAREN, 0)

        def REGISTER(self):
            return self.getToken(UniAssemblyParser.REGISTER, 0)

        def RPAREN(self):
            return self.getToken(UniAssemblyParser.RPAREN, 0)

        def LBRACK(self):
            return self.getToken(UniAssemblyParser.LBRACK, 0)

        def COMMA(self):
            return self.getToken(UniAssemblyParser.COMMA, 0)

        def relocation_expr(self):
            return self.getTypedRuleContext(UniAssemblyParser.Relocation_exprContext,0)


        def RBRACK(self):
            return self.getToken(UniAssemblyParser.RBRACK, 0)

        def IMMEDIATE(self):
            return self.getToken(UniAssemblyParser.IMMEDIATE, 0)

        def BANG(self):
            return self.getToken(UniAssemblyParser.BANG, 0)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_memory_operand

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMemory_operand" ):
                listener.enterMemory_operand(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMemory_operand" ):
                listener.exitMemory_operand(self)




    def memory_operand(self):

        localctx = UniAssemblyParser.Memory_operandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_memory_operand)
        try:
            self.state = 113
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,8,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 81
                self.match(UniAssemblyParser.NUMBER)
                self.state = 82
                self.match(UniAssemblyParser.LPAREN)
                self.state = 83
                self.match(UniAssemblyParser.REGISTER)
                self.state = 84
                self.match(UniAssemblyParser.RPAREN)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 85
                self.match(UniAssemblyParser.LPAREN)
                self.state = 86
                self.match(UniAssemblyParser.REGISTER)
                self.state = 87
                self.match(UniAssemblyParser.RPAREN)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 88
                self.match(UniAssemblyParser.LBRACK)
                self.state = 89
                self.match(UniAssemblyParser.REGISTER)
                self.state = 90
                self.match(UniAssemblyParser.COMMA)
                self.state = 91
                self.relocation_expr()
                self.state = 92
                self.match(UniAssemblyParser.RBRACK)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 94
                self.match(UniAssemblyParser.LBRACK)
                self.state = 95
                self.match(UniAssemblyParser.REGISTER)
                self.state = 96
                self.match(UniAssemblyParser.RBRACK)
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 97
                self.match(UniAssemblyParser.LBRACK)
                self.state = 98
                self.match(UniAssemblyParser.REGISTER)
                self.state = 99
                self.match(UniAssemblyParser.COMMA)
                self.state = 100
                self.match(UniAssemblyParser.NUMBER)
                self.state = 101
                self.match(UniAssemblyParser.RBRACK)
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 102
                self.match(UniAssemblyParser.LBRACK)
                self.state = 103
                self.match(UniAssemblyParser.REGISTER)
                self.state = 104
                self.match(UniAssemblyParser.COMMA)
                self.state = 105
                self.match(UniAssemblyParser.IMMEDIATE)
                self.state = 106
                self.match(UniAssemblyParser.RBRACK)
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 107
                self.match(UniAssemblyParser.LBRACK)
                self.state = 108
                self.match(UniAssemblyParser.REGISTER)
                self.state = 109
                self.match(UniAssemblyParser.COMMA)
                self.state = 110
                self.match(UniAssemblyParser.NUMBER)
                self.state = 111
                self.match(UniAssemblyParser.RBRACK)
                self.state = 112
                self.match(UniAssemblyParser.BANG)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Modified_symbolContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def COLON(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.COLON)
            else:
                return self.getToken(UniAssemblyParser.COLON, i)

        def SYMBOL(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.SYMBOL)
            else:
                return self.getToken(UniAssemblyParser.SYMBOL, i)

        def DOT_SYMBOL(self):
            return self.getToken(UniAssemblyParser.DOT_SYMBOL, 0)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_modified_symbol

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterModified_symbol" ):
                listener.enterModified_symbol(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitModified_symbol" ):
                listener.exitModified_symbol(self)




    def modified_symbol(self):

        localctx = UniAssemblyParser.Modified_symbolContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_modified_symbol)
        try:
            self.state = 123
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 115
                self.match(UniAssemblyParser.COLON)
                self.state = 116
                self.match(UniAssemblyParser.SYMBOL)
                self.state = 117
                self.match(UniAssemblyParser.COLON)
                self.state = 118
                self.match(UniAssemblyParser.SYMBOL)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 119
                self.match(UniAssemblyParser.COLON)
                self.state = 120
                self.match(UniAssemblyParser.SYMBOL)
                self.state = 121
                self.match(UniAssemblyParser.COLON)
                self.state = 122
                self.match(UniAssemblyParser.DOT_SYMBOL)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Relocation_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def HASH(self):
            return self.getToken(UniAssemblyParser.HASH, 0)

        def COLON(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.COLON)
            else:
                return self.getToken(UniAssemblyParser.COLON, i)

        def SYMBOL(self, i:int=None):
            if i is None:
                return self.getTokens(UniAssemblyParser.SYMBOL)
            else:
                return self.getToken(UniAssemblyParser.SYMBOL, i)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_relocation_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRelocation_expr" ):
                listener.enterRelocation_expr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRelocation_expr" ):
                listener.exitRelocation_expr(self)




    def relocation_expr(self):

        localctx = UniAssemblyParser.Relocation_exprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_relocation_expr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 125
            self.match(UniAssemblyParser.HASH)
            self.state = 126
            self.match(UniAssemblyParser.COLON)
            self.state = 127
            self.match(UniAssemblyParser.SYMBOL)
            self.state = 128
            self.match(UniAssemblyParser.COLON)
            self.state = 129
            self.match(UniAssemblyParser.SYMBOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DOT(self):
            return self.getToken(UniAssemblyParser.DOT, 0)

        def MINUS(self):
            return self.getToken(UniAssemblyParser.MINUS, 0)

        def SYMBOL(self):
            return self.getToken(UniAssemblyParser.SYMBOL, 0)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_expression

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExpression" ):
                listener.enterExpression(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExpression" ):
                listener.exitExpression(self)




    def expression(self):

        localctx = UniAssemblyParser.ExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_expression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 131
            self.match(UniAssemblyParser.DOT)
            self.state = 132
            self.match(UniAssemblyParser.MINUS)
            self.state = 133
            self.match(UniAssemblyParser.SYMBOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class OperandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REGISTER(self):
            return self.getToken(UniAssemblyParser.REGISTER, 0)

        def segmented_memory_operand(self):
            return self.getTypedRuleContext(UniAssemblyParser.Segmented_memory_operandContext,0)


        def memory_operand(self):
            return self.getTypedRuleContext(UniAssemblyParser.Memory_operandContext,0)


        def modified_symbol(self):
            return self.getTypedRuleContext(UniAssemblyParser.Modified_symbolContext,0)


        def expression(self):
            return self.getTypedRuleContext(UniAssemblyParser.ExpressionContext,0)


        def NUMBER(self):
            return self.getToken(UniAssemblyParser.NUMBER, 0)

        def IMMEDIATE(self):
            return self.getToken(UniAssemblyParser.IMMEDIATE, 0)

        def BINARY_IMMEDIATE(self):
            return self.getToken(UniAssemblyParser.BINARY_IMMEDIATE, 0)

        def SYMBOL(self):
            return self.getToken(UniAssemblyParser.SYMBOL, 0)

        def DOT_SYMBOL(self):
            return self.getToken(UniAssemblyParser.DOT_SYMBOL, 0)

        def STRING(self):
            return self.getToken(UniAssemblyParser.STRING, 0)

        def getRuleIndex(self):
            return UniAssemblyParser.RULE_operand

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOperand" ):
                listener.enterOperand(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOperand" ):
                listener.exitOperand(self)




    def operand(self):

        localctx = UniAssemblyParser.OperandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_operand)
        try:
            self.state = 146
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,10,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 135
                self.match(UniAssemblyParser.REGISTER)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 136
                self.segmented_memory_operand()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 137
                self.memory_operand()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 138
                self.modified_symbol()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 139
                self.expression()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 140
                self.match(UniAssemblyParser.NUMBER)
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 141
                self.match(UniAssemblyParser.IMMEDIATE)
                pass

            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 142
                self.match(UniAssemblyParser.BINARY_IMMEDIATE)
                pass

            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 143
                self.match(UniAssemblyParser.SYMBOL)
                pass

            elif la_ == 10:
                self.enterOuterAlt(localctx, 10)
                self.state = 144
                self.match(UniAssemblyParser.DOT_SYMBOL)
                pass

            elif la_ == 11:
                self.enterOuterAlt(localctx, 11)
                self.state = 145
                self.match(UniAssemblyParser.STRING)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





