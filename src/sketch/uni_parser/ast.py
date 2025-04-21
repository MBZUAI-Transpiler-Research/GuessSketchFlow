from dataclasses import dataclass, field
from src.sketch.uni_parser.UniAssemblyListener import UniAssemblyListener
from typing import List, Union


@dataclass
class Operand:
    pass


@dataclass
class Register(Operand):
    name: str


@dataclass
class Immediate(Operand):
    value: int


@dataclass
class Symbol(Operand):
    name: str


@dataclass
class DotSymbol(Operand):
    name: str


@dataclass
class RelocationExpr(Operand):
    modifier: str
    symbol: str


@dataclass
class ModifiedSymbol(Operand):
    modifier: str
    symbol: str


@dataclass
class Expression(Operand):  # Represents . - symbol
    base: str
    minus: str


@dataclass
class MemoryOperand(Operand):
    base: str
    offset: Union[int, RelocationExpr, None]
    preindexed: bool = False


@dataclass
class SegmentedMemoryOperand(Operand):
    base: str
    offset: str


@dataclass
class Instruction:
    name: str
    operands: List[Operand]


@dataclass
class Directive:
    name: str
    operands: List[Operand]
    scope_chain: List[str] = field(default_factory=list)


@dataclass
class Label:
    name: str
    scope_chain: List[str] = field(default_factory=list)


class ASTListener(UniAssemblyListener):
    def __init__(self):
        self.nodes = []
        self._current_operands = []

    def _push_operand(self, operand):
        self._current_operands.append(operand)

    def _flush_operands(self):
        ops = self._current_operands
        self._current_operands = []
        return ops

    def exitInstruction(self, ctx):
        name = ctx.SYMBOL().getText()
        operands = self._flush_operands()
        self.nodes.append(Instruction(name, operands))

    def exitDirective(self, ctx):
        name = ctx.DOT_SYMBOL().getText()
        scoped = [s.getText() for s in ctx.SYMBOL()] if ctx.SYMBOL() else []
        operands = self._flush_operands()
        directive = Directive(name=name, operands=operands)

        if scoped:
            directive.scope_chain = scoped

        self.nodes.append(directive)

    def exitLabel(self, ctx):
        scoped = [s.getText() for s in ctx.SYMBOL()] if ctx.SYMBOL() else []

        if ctx.DOT_SYMBOL():
            name = ctx.DOT_SYMBOL().getText()
        else:
            name = ctx.SYMBOL()[-1].getText()

        label = Label(name=name)

        if scoped:
            label.scope_chain = scoped

        self.nodes.append(label)

    def _handle_memory(self, ctx):
        base = ctx.REGISTER().getText()
        if ctx.NUMBER():
            offset = int(ctx.NUMBER().getText())
        elif ctx.IMMEDIATE():
            offset = int(ctx.IMMEDIATE().getText()[1:])
        elif ctx.relocation_expr():
            parts = [t.getText() for t in ctx.relocation_expr().getChildren()]
            offset = RelocationExpr(modifier=parts[2], symbol=parts[4])
        else:
            offset = None

        preindexed = ctx.BANG() is not None
        self._push_operand(
            MemoryOperand(
                base=base, offset=offset, preindexed=preindexed
            )
        )

    def exitRelocation_expr(self, ctx):
        modifier = ctx.SYMBOL(0).getText()
        symbol = ctx.SYMBOL(1).getText()
        self._push_operand(RelocationExpr(modifier=modifier, symbol=symbol))

    def exitOperand(self, ctx):
        text = ctx.getText()

        if ctx.REGISTER():
            self._push_operand(Register(name=text))

        elif ctx.NUMBER():
            self._push_operand(Immediate(value=int(text)))

        elif ctx.IMMEDIATE():
            value = int(ctx.IMMEDIATE().getText()[1:])  # strip leading #
            self._push_operand(Immediate(value=value))

        elif ctx.STRING():
            self._push_operand(Symbol(name=text.strip('"')))

        elif ctx.modified_symbol():
            parts = [t.getText() for t in ctx.modified_symbol().getChildren()]
            self._push_operand(ModifiedSymbol(
                modifier=parts[1], symbol=parts[3]))

        elif ctx.expression():
            base = ctx.expression().DOT().getText()
            minus = ctx.expression().SYMBOL().getText()
            self._push_operand(Expression(base=base, minus=minus))

        elif ctx.memory_operand():
            self._handle_memory(ctx.memory_operand())

        elif ctx.segmented_memory_operand():
            parts = [t.getText()
                     for t in ctx.segmented_memory_operand().getChildren()]
            self._push_operand(
                SegmentedMemoryOperand(
                    base=parts[0], offset=parts[2]
                )
            )

        elif ctx.SYMBOL():
            self._push_operand(Symbol(name=text))

        elif ctx.DOT_SYMBOL():
            self._push_operand(DotSymbol(name=text))
