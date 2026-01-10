#!/usr/bin/env python3
""" Plot a histogram """
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ Histogram with bins of size 10 """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.ylim(0, 30)
    bounds = np.arange(0, 101, 10)
    plt.xticks(bounds)
    plt.hist(student_grades, range=(0, 100), bins=bounds, edgecolor="black")
