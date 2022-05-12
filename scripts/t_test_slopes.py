#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Run a t-test on the slope coefficients to measure the statistical significance of the observed
"amortisation" of the loss, given two sets of errors (baseline vs meta-learner).
"""


import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go

from typing import List
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from plotly.tools import FigureFactory as FF

import pandas as pd
import numpy as np
import argparse
import scipy
import csv
import sys


def parse_file(f_path: str):
    indices, data = [], []
    with open(f_path, "r") as f:
        reader = csv.reader(f)
        header = None
        for line in reader:
            if header is None:
                header = line
                continue
            data.append(float(line[1]))
            indices.append(int(line[0])),
    return indices, np.array(data)

def merged(file_1: str, file_2: str):
    indices1, data1 = parse_file(file_1)
    indices2, data2 = parse_file(file_2)
    assert indices1 == indices2
    slope1, intercept1, r_value_1, p_value_1, std_err_1 = stats.linregress(np.array(list(range(data1.shape[0]))), data1)
    slope2, intercept2, r_value_2, p_value_2, std_err_2 = stats.linregress(np.array(list(range(data2.shape[0]))), data2)
    print("[*] Merged files")
    print(f"[*] Slope of data1: {slope1} - Slope of data2: {slope2}\n-> Difference: {slope1-slope2}")

    print("[*] Computing statistical significance via Student's t-test on a regression model with"
            " an interaction term...")
    df = pd.DataFrame(data=data1, columns=['error'])
    df['level'] = indices1 # Covariates
    df['method'] = file_1[file_1.rfind('/')+1:].split('.')[0].split('_')[0] # Condition variable
    df2 = pd.DataFrame(data=data2, columns=['error'])
    df2['method'] = file_2[file_2.rfind('/')+1:].split('.')[0].split('_')[0] # Condition variable
    df2['level'] = indices2 # Covariates
    # I joined the two data frames into one, and added a condition value ('method') for each model.
    # Now I only need to fit a regression model (or ANOVA?) that includes an interaction term:
    # condition*X. That way, I can determine whether the coefficient for X depends on the
    # condition: I want to measure the interaction effect of condition onto X for the dependent
    # variable Y.
    df = pd.concat([df, df2])
    model = ols("error ~ level*method", df).fit()
    pval = model.pvalues['level:method[T.baseline]']
    print(model.summary())
    # A high value of t implies a high significance for the coefficient. The p value, P > |t|, is a
    # measurement of how likely the coefficient is measured through the model by chance. The common
    # threshold is 0.05, meaning that a p value above that threshold signifies that the coefficient
    # has little to no impact on the dependent variable.
    print(f"\n\n[*] P-value: {pval}")
    print(f"[*] Null hypothesis rejected? {'YES' if pval < 0.05 else 'NO'}")
    # Now run an ANalysis Of VAriance to compare the joint regression model against the conditionned
    # regression model
    anova_results = anova_lm(ols("error ~ level", df).fit(), model)
    print("\n\n[*] ANOVA:")
    print(anova_results)
    fig = px.scatter(df, x="level", y="error", color="method", trendline="ols")
    fig.show()
    pio.kaleido.scope.mathjax = None
    fig.write_image("fig.pdf")


def normal(file_1: str, file_2: str):
    _, data1 = parse_file(file_1)
    _, data2 = parse_file(file_2)
    slope1, intercept1, r_value_1, p_value_1, std_err_1 = stats.linregress(np.array(list(range(data1.shape[0]))), data1)
    slope2, intercept2, r_value_2, p_value_2, std_err_2 = stats.linregress(np.array(list(range(data2.shape[0]))), data2)
    print(f"[*] Slope of data1: {slope1} - Slope of data2: {slope2}\n-> Difference: {slope1-slope2}")

    print("[*] Computing statistical significance via Student's t-test on a regression model with"
            " an interaction term...")
    df = pd.DataFrame(data=data1, columns=['error'])
    df['level'] = list(range(data1.shape[0])) # Covariates
    df['method'] = file_1[file_1.rfind('/')+1:].split('.')[0].split('_')[0] # Condition variable
    df2 = pd.DataFrame(data=data2, columns=['error'])
    df2['method'] = file_2[file_2.rfind('/')+1:].split('.')[0].split('_')[0] # Condition variable
    df2['level'] = list(range(data2.shape[0])) # Covariates
    # I joined the two data frames into one, and added a condition value ('method') for each model.
    # Now I only need to fit a regression model (or ANOVA?) that includes an interaction term:
    # condition*X. That way, I can determine whether the coefficient for X depends on the
    # condition: I want to measure the interaction effect of condition onto X for the dependent
    # variable Y.
    df = pd.concat([df, df2])
    print(df)
    model = ols("error ~ level*method", df).fit()
    pval = model.pvalues['level:method[T.baseline]']
    print(model.summary())
    # A high value of t implies a high significance for the coefficient. The p value, P > |t|, is a
    # measurement of how likely the coefficient is measured through the model by chance. The common
    # threshold is 0.05, meaning that a p value above that threshold signifies that the coefficient
    # has little to no impact on the dependent variable.
    print(f"\n\n[*] P-value: {pval}")
    print(f"[*] Null hypothesis rejected? {'YES' if pval < 0.05 else 'NO'}")
    # Now run an ANalysis Of VAriance to compare the joint regression model against the conditionned
    # regression model
    anova_results = anova_lm(ols("error ~ level", df).fit(), model)
    print("\n\n[*] ANOVA:")
    print(anova_results)
    fig = px.scatter(df, x="level", y="error", color="method", trendline="ols")
    fig.show()
    pio.kaleido.scope.mathjax = None
    fig.write_image("fig.pdf")

def normal_2(file_1: str, file_2: str):
    '''
    Alternative way of doing it, I'm not sure what is more suited... The pvalues aren't the same
    but they all reject the null hypothesis for the same cases.
    '''
    _, data1 = parse_file(file_1)
    _, data2 = parse_file(file_2)
    slope1, intercept1, r_value_1, p_value_1, std_err_1 = stats.linregress(np.array(list(range(data1.shape[0]))), data1)
    slope2, intercept2, r_value_2, p_value_2, std_err_2 = stats.linregress(np.array(list(range(data2.shape[0]))), data2)
    print(f"[*] Slope of data1: {slope1} - Slope of data2: {slope2}\n-> Difference: {slope1-slope2}")

    print("[*] Computing statistical significance via Student's t-test on a regression model with"
            " an interaction term...")
    df = pd.DataFrame(data=data1, columns=['error'])
    df['level'] = list(range(data1.shape[0])) # Covariates
    df['method'] = 0 # Condition variable
    df2 = pd.DataFrame(data=data2, columns=['error'])
    df2['method'] = 1 # Condition variable
    df2['level'] = list(range(data2.shape[0])) # Covariates
    # I joined the two data frames into one, and added a condition value ('method') for each model.
    # Now I only need to fit a regression model (or ANOVA?) that includes an interaction term:
    # condition*X. That way, I can determine whether the coefficient for X depends on the
    # condition: I want to measure the interaction effect of condition onto X for the dependent
    # variable Y.
    df = pd.concat([df, df2])
    df['int'] = df['level'] * df['method'] # Interaction term
    print(df)
    model = ols("error ~ level + method + int", df).fit()
    # pval = model.pvalues['level:method[T.baseline]']
    print(model.summary())
    # A high value of t implies a high significance for the coefficient. The p value, P > |t|, is a
    # measurement of how likely the coefficient is measured through the model by chance. The common
    # threshold is 0.05, meaning that a p value above that threshold signifies that the coefficient
    # has little to no impact on the dependent variable.
    hypotheses = '(int = 0), (method = 0)'
    f_test = model.f_test(hypotheses)
    print(f_test)
    print(f"\n\n[*] P-value: {f_test.pvalue}")
    print(f"[*] Null hypothesis rejected? {'YES' if f_test.pvalue < 0.05 else 'NO'}")
    # Now run an ANalysis Of VAriance to compare the joint regression model against the conditionned
    # regression model
    anova_results = anova_lm(ols("error ~ level", df).fit(), model)
    print("\n\n[*] ANOVA:")
    print(anova_results)
    fig = px.scatter(df, x="level", y="error", color="method", trendline="ols")
    fig.show()
    pio.kaleido.scope.mathjax = None
    fig.write_image("fig.pdf")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file_1", type=str)
    parser.add_argument("csv_file_2", type=str)
    parser.add_argument("--merged", action='store_true', help="Whether this file is the result of a merge")
    args = parser.parse_args()
    if args.merged:
        merged(args.csv_file_1, args.csv_file_2)
    else:
        normal(args.csv_file_1, args.csv_file_2)

