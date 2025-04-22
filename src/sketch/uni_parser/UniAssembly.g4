grammar UniAssembly;

// === Parser rules ===

program
    : line (NEWLINE line)* EOF
    ;

line
    : instruction
    | label
    | directive
    ;

instruction
    : SYMBOL operands?
    ;

label
    : (LCURLY SYMBOL RCURLY)* DOT_SYMBOL COLON
    | SYMBOL COLON
    ;

directive
    : (LCURLY SYMBOL RCURLY)* DOT_SYMBOL operands?
    ;

// === Operand rules ===

operands
    : operand (COMMA operand)*
    ;

segmented_memory_operand
    : REGISTER COLON NUMBER
    ;

memory_operand
    : NUMBER LPAREN REGISTER RPAREN
    | LPAREN REGISTER RPAREN
    | LBRACK REGISTER COMMA relocation_expr RBRACK
    | LBRACK REGISTER RBRACK   
    | LBRACK REGISTER COMMA NUMBER RBRACK
    | LBRACK REGISTER COMMA IMMEDIATE RBRACK
    | LBRACK REGISTER COMMA NUMBER RBRACK BANG
    ;

modified_symbol
    : COLON SYMBOL COLON SYMBOL
    | COLON SYMBOL COLON DOT_SYMBOL
    ;

relocation_expr
    : HASH COLON SYMBOL COLON SYMBOL
    ;

expression
    : DOT MINUS SYMBOL
    ;

operand
    : REGISTER
    | segmented_memory_operand
    | memory_operand
    | modified_symbol
    | expression
    | NUMBER
    | IMMEDIATE
    | BINARY_IMMEDIATE
    | SYMBOL
    | DOT_SYMBOL
    | STRING
    ;

REGISTER: 'x' [0-9]+ | 'r' [0-9]+ | 'a' [0-9]+ | 's' [0-9]+ | 'w' [0-9]+ | 'sp' | '%' (
    // 64-bit general-purpose registers
    'r' ( 'ax' | 'bx' | 'cx' | 'dx' | 'si' | 'di' | 'sp' | 'bp' )
    | 'r' [8-9]
    | 'r' '1' [0-5]
    // 32-bit general-purpose registers
    | 'e' ( 'ax' | 'bx' | 'cx' | 'dx' | 'si' | 'di' | 'sp' | 'bp' )
    // 16-bit general-purpose registers
    | 'a' 'x'
    | 'b' 'x'
    | 'c' 'x'
    | 'd' 'x'
    | 's' 'i'
    | 'd' 'i'
    | 's' 'p'
    | 'b' 'p'
    // 8-bit general-purpose registers
    | 'a' 'l'
    | 'b' 'l'
    | 'c' 'l'
    | 'd' 'l'
    | 'a' 'h'
    | 'b' 'h'
    | 'c' 'h'
    | 'd' 'h'
    | 'f' 's'
);

SYMBOL: [a-zA-Z_@%][a-zA-Z0-9_@.-]*;

DOT_SYMBOL: '.'? [a-zA-Z_][a-zA-Z0-9_@.-]*;

IMMEDIATE: '#' [0-9]+ | '$' [0-9]+;
BINARY_IMMEDIATE: '#0x' [0-9]+;
NUMBER: '-'? [0-9]+;

STRING: '"' (~["\r\n] | '\\' .)* '"';

COLON: ':';
COMMA: ',';

LPAREN: '(';
RPAREN: ')';

LBRACK: '[';
RBRACK: ']';
HASH: '#';

BANG: '!';

DOT: '.';
MINUS: '-';

LCURLY: '{';
RCURLY: '}';

NEWLINE: '\r'? '\n';
WS: [ \t]+ -> skip;
