# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:09:14 2021

@author: Alexandre
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def default_gauge(default_proba):
    
    if default_proba <= 20:
        color = 'green'
    elif default_proba <= 40:
        color = 'green'
    elif default_proba <= 60:
        color = 'yellow'
    elif default_proba <= 80:
        color = 'orange'
    else:
        color = 'red'
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = default_proba,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilité de défaut de paiement (%)"},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': color}}))
    
    return fig


def plotly_waterfall(sk_id, shap_series, n_feats=10):
    
    shap_series = pd.DataFrame(shap_series)
    shap_series = shap_series.reindex(shap_series[str(sk_id)].abs().sort_values().index)
    shap_series = shap_series[str(sk_id)]

    values = list(shap_series.values*100)
    first_val = sum(values[:(381-n_feats)])
    values = [first_val] + values[381-n_feats:]
    text = [ '%.2f' % elem for elem in values ]
    for i,txt in enumerate(text):
        if float(txt) > 0:
            text[i] = '+' + txt
    text = [txt + ' %' for txt in text]
    ticks = shap_series.index
    ticks = ['Autres'] + list(ticks[381-n_feats:])
    fig = go.Figure(go.Waterfall(
        orientation='h',
        x = values,
        y= ticks, base = 50,
        text = text,
        measure = ["relative" for k in range(len(text)+1)],
        increasing = {"marker":{"color":"red", "line":{"color":"#C10000", "width":2}}},
        decreasing = {"marker":{"color":"green", "line":{"color":"#2B9B00", "width":2}}},
    ))

    fig.update_layout(title = "Influence de chaque varibale sur la prédiction de difficulté de paiement", waterfallgap = 0.2)
    fig.update_layout(height=int(n_feats*50))
    VARS = ticks[1:]
    VARS.reverse()
    
    return fig, VARS


def plot_bar_default(df, var, sk_id, bins=np.array([]), n_bins=10):
    
    value = round(df.loc[sk_id][var],2)
    
    var_data = df[['TARGET', var]]
    min_var, max_var = round(df[var].min(),1), round(df[var].max(),1)
    
    if bins.any()==False:
        bins = np.linspace(min_var, max_var, num = n_bins)
    
    var_data['var_bin'] = pd.cut(var_data[var], bins = bins)
    var_groups  = var_data.groupby('var_bin').mean()
    
    
    y = 100 * var_groups['TARGET']
    
    x = var_groups.index.astype(str)
    x = [string.replace(",", " -") for string in x]
    x = [string.replace("(", "") for string in x]
    x = [string.replace("]", "") for string in x]
    
    colors = ['red' for i in range(len(y))]
    i=0
    while value not in var_groups.index[i]:
        i+=1
    colors[i] = 'purple'
    
    fig = go.Figure([go.Bar(x=x, y=y, marker_color=colors)])
    fig.update_layout(title=f'Taux de difficulté de paiement (%) par {var} \
    (Client du prêt n°{sk_id} en violet)')
    
    return fig

def plot_bars(df, var, sk_id):

    default_tab = df[df.TARGET==1].groupby(var)['TARGET'].count()
    total_tab = df.groupby(var)['TARGET'].count()

    ind = total_tab.index
    default = []
    
    for i, index in enumerate(total_tab.index):
        default.append(default_tab[index]/total_tab[index]*100)
     
    colors=['red', 'red']
    value = df.loc[sk_id][var]
    i=0
    while value != total_tab.index[i]:
        i+=1
    colors[i] = 'purple'
            
    fig = go.Figure([go.Bar(x=ind, y=default, marker_color=colors)])
    fig.update_layout(title=f'Taux de difficulté de paiement (%) par {var} \
    (Client du prêt n°{sk_id} en violet)')
    
    return fig, value
