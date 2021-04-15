import pathlib
from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

LANG_TO_INT = {
        "C": 1,
        "C++": 2,
        "Go": 3,
        "Haskell": 4,
        "Java": 5,
        "JavaScript": 6,
        "Julia": 7,
        "Perl": 8,
        "Python": 9,
        "Ruby": 10
}

INT_TO_LANG = {i: lang for (lang, i) in LANG_TO_INT.items()}

EXT_TO_LANG = {
    ".c": "C",
    ".cpp": "C++",
    ".go": "Go",
    ".hs": "Haskell",
    ".java": "Java",
    ".js": "JavaScript",
    ".jl": "Julia",
    ".pl": "Perl",
    ".py": "Python",
    ".rb": "Ruby"
}

LANG_TO_EXT = {l: e for e, l in EXT_TO_LANG.items()}


def get_language_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    return EXT_TO_LANG[ext]


def get_path_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    language = EXT_TO_LANG[ext]
    return f'../lang/{language}/{filename}'


SPECIAL_CHAR_NAMES = {"'": "squote", "~": "tilde", "`": "backtick", "!": "exclaim", "@": "at", "#": "pound",
                          "$": "dollar", "%": "pct", "^": "caret", "&": "amp", "*": "times", "(": "lparen",
                          ")": "rparen", "-": "minus", "+": "plus", "=": "eq", "[": "lbracket", "]": "rbracket",
                          "{": "lbrace", "}": "rbrace", "|": "pipe", ";": "semicolon", ":": "colon", '"': "dquote",
                          ",": "comma", ".": "dot", "<": "langle", ">": "rangle", "/": "fslash", "?": "question",
                          "\\": "bslash", "...": "ellipsis"}

SPECIAL_CHARS = list(SPECIAL_CHAR_NAMES.keys())

CHAR_MAPPING = {"'": 0, '~': 1, '`': 2, '!': 3, '@': 4, '#': 5, '$': 6, '%': 7, '^': 8, '&': 9, '*': 10, '(': 11,
                ')': 12, '-': 13, '+': 14, '=': 15, '[': 16, ']': 17, '{': 18, '}': 19, '|': 20, ';': 21, ':': 22,
                '"': 23, ',': 24, '.': 25, '<': 26, '>': 27, '/': 28, '?': 29, '\\': 30, '...': 31, 'a': 32,
                'b': 33,
                'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39, 'i': 40, 'j': 41, 'k': 42, 'l': 43, 'm': 44,
                'n': 45, 'o': 46, 'p': 47, 'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54, 'x': 55,
                'y': 56, 'z': 57, '_': 58, '1': 59, '2': 60, '3': 61, '4': 62, '5': 63, '6': 64, '7': 65, '8': 66,
                '9': 67, '0': 68}

DEFAULT_KEYWORDS = ['__end__', 'and', 'any', 'as', 'assert', 'auto', 'begin', 'bool', 'boolean', 'break', 'byte',
                    'case', 'catch', 'char', 'check', 'class', 'const', 'continue', 'cout', 'data', 'def',
                    'default', 'delete', 'deriving', 'do', 'double', 'elif', 'else', 'elseif', 'elsif', 'end',
                    'endl', 'error', 'eval', 'except', 'export', 'extends', 'extern', 'false', 'final', 'float64',
                    'for', 'foreach', 'from', 'func', 'function', 'go', 'goto', 'if', 'implements', 'import', 'in',
                    'include', 'instance', 'int', 'int64', 'interface', 'iostream', 'is', 'lambda', 'last', 'let',
                    'local', 'long', 'main', 'map', 'module', 'my', 'namespace', 'new', 'next', 'nil', 'none',
                    'not', 'nothing', 'null', 'of', 'operator', 'or', 'our', 'package', 'print', 'private',
                    'public', 'qualified', 'raise', 'range', 'return', 'self', 'sizeof', 'static', 'std', 'string',
                    'struct', 'switch', 'template', 'then', 'this', 'thread', 'throw', 'throws', 'true', 'try',
                    'type', 'typedef', 'typename', 'typeof', 'undef', 'union', 'unless', 'unsigned', 'use', 'using',
                    'var', 'void', 'when', 'where', 'while', 'with']

RANDOM_SEED = 12345678


Model = Union[DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, SGDClassifier, GaussianNB,
              MLPClassifier, SVC, NuSVC, LinearSVC, MultinomialNB, KNeighborsClassifier, BernoulliNB]
