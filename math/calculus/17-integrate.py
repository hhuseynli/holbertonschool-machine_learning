#!/usr/bin/env python3
""" Integrate coefficents """


def poly_integral(poly, C=0):
    """ Integration of polynomials """
    if type(poly) is not list:
        return None
    elif not poly:
        return None
    elif len(poly) == 0:
        return None
    inte = [C]
    for i in range(len(poly)):
        coe = poly[i] / (i+1)
        if int(coe) == coe:
            inte.append(int(coe))
        else:
            inte.append(coe)
    return inte
