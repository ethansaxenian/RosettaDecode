import os
import pathlib


RESERVED_KEYWORDS = ['abstract', 'and', 'as', 'assert', 'begin', 'bool', 'boolean', 'break', 'byte', 'case', 'catch',
                     'chan', 'char', 'class', 'const', 'continue', 'def', 'default', 'defer', 'del', 'delete',
                     'deriving', 'die', 'do', 'double', 'elif', 'else', 'elseif', 'elsif', 'end', 'enum', 'eq', 'eval',
                     'except', 'exit', 'extends', 'false', 'final', 'finally', 'float', 'for', 'foreach', 'friend',
                     'from', 'func', 'function', 'global', 'go', 'goto', 'if', 'implements', 'import', 'in', 'inline',
                     'instanceof', 'int', 'interface', 'is', 'lambda', 'let', 'local', 'long', 'map', 'module', 'my',
                     'namespace', 'new', 'nil', 'none', 'not', 'null', 'of', 'operator', 'or', 'package',
                     'pass', 'print', 'private', 'proc', 'protected', 'public', 'qualified', 'raise', 'range', 'ref',
                     'require', 'rescue', 'return', 'self', 'short', 'signed', 'sizeof', 'static', 'struct',
                     'super', 'switch', 'template', 'then', 'this', 'throw', 'throws', 'true', 'try', 'type', 'typedef',
                     'typename', 'typeof', 'undef', 'undefined', 'unless', 'unsigned', 'until', 'use', 'using', 'var',
                     'virtual', 'void', 'when', 'where', 'while', 'with', 'yield']


EXTENSION_TO_LANGUAGE = {
    ".c": "C",
    ".cpp": "C++",
    ".go": "Go",
    ".hs": "Haskell",
    ".java": "Java",
    ".js": "JavaScript",
    ".julia": "Julia",
    ".pl": "Perl",
    ".py": "Python",
    ".rb": "Ruby"
}

LANGUAGE_FILES = [file for lang in EXTENSION_TO_LANGUAGE.values() for file in os.listdir(f'lang/{lang}')]


def get_path_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    language = EXTENSION_TO_LANGUAGE[ext]
    return f'lang/{language}/{filename}'


def get_language_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    return EXTENSION_TO_LANGUAGE[ext]
