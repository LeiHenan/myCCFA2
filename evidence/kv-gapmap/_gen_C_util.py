# -*- coding: utf-8 -*-
def R(k):
    """quote/URL pulled verbatim from row k of _in_C.tsv"""
    return ('r', k)

def T(s):
    """literal text (verifier-corrected wording)"""
    return ('t', s)
