import tokenize
from io import StringIO

def is_variable( exp):
    return isinstance( exp, str) and exp[ 0] == "?"

def is_constant( exp):
    return isinstance( exp, str) and not is_variable( exp)

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def occurs_check( exp1, exp2):
    return exp1 in flatten( exp2)

def inconsistent_assignment( exp1, exp2, frame):
    if not exp1 in frame:
        return False
    return not frame[ exp1] == exp2

def unification( exp1, exp2, frame=None):

    if frame == None:
        frame = {}
    if is_constant( exp1) and is_constant( exp2) or len( exp1) == 0 and len( exp2) == 0:
        if exp1 == exp2:
            return frame
        else:
            return False
    if is_variable( exp1):
        if occurs_check( exp1, exp2) or inconsistent_assignment( exp1, exp2, frame):
            return False
        else:
            frame[ exp1] = exp2
            return frame
    if is_variable( exp2):
        if occurs_check( exp2, exp1) or inconsistent_assignment( exp2, exp1, frame):
            return False
        else:
            frame[ exp2] = exp1
            return frame
    head1 = exp1[ 0]
    head2 = exp2[ 0]
    frame = unification( head1, head2, frame)
    if frame == False:
        return False
    return unification( exp1[1:], exp2[1:], frame)

import sys
import tokenize
from io import StringIO

def atom(next, token):
    if token[1] == '(':
        out = []
        token = next()
        while token[1] != ')':
            out.append(atom( next, token))
            token = next()
            if token[1] == ' ':
                token = next()
        return out
    elif token[1] == '?':
        token = next()
        return "?" + token[1]
    else:
        return token[1]

def parse(exp):
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())

def unify( exp1, exp2):
    return unification( parse( exp1), parse( exp2))
