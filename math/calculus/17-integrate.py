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
    elif type(C) is not int:
        return None
    elif poly == [0]:
        return [C]

    integral = [C]
    for i in range(len(poly)):
        x = poly[i] / (i + 1)
        integral.append(int(x) if x.is_integer() else x)
    return integral
