from src.sketch.uni_parser.UniAssemblyParser import UniAssemblyParser
from src.sketch.uni_parser.ast import ASTListener
from src.sketch.uni_parser.UniAssemblyLexer import UniAssemblyLexer
from antlr4 import *
from antlr4.error.ErrorStrategy import BailErrorStrategy
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import ParseCancellationException


class LexerBailErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise ParseCancellationException(
            f"Lexer error at {line}:{column}: {msg}"
        )


def parse_assembly(input_text: str):
    cleaned_text = input_text.split(";")[0].strip()
    cleaned_text = cleaned_text.split("##")[0].strip()
    stream = InputStream(cleaned_text)

    lexer = UniAssemblyLexer(stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(LexerBailErrorListener())

    tokens = CommonTokenStream(lexer)

    parser = UniAssemblyParser(tokens)
    parser._errHandler = BailErrorStrategy()
    parser.removeErrorListeners()

    try:
        tree = parser.program()

        walker = ParseTreeWalker()
        listener = ASTListener()
        walker.walk(listener, tree)

        return listener.nodes
    except ParseCancellationException as e:
        return None
