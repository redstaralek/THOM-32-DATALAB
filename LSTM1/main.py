import math, os, gc, sys, joblib, pandas as pd, csv, numpy as np, tensorflow as tf
from timeseries_rnn import *
from estacao import *
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--treina', default='True')
    return parser

def main(args):
    X, Y = [], []
    with open('clima_bsb.csv') as _csv:
        _r = csv.reader(_csv, delimiter=',')
        linha= 0
        linha_max = 20*365*24

        for row in _r:
            if linha > 0:
                ano = row[0]
                horario = int(float(row[2]))
                dia_ano_str = row[1]
                dia_ano = int(float(dia_ano_str))

                dia_ano_str.rjust(3 + len(dia_ano_str), '0')
                data = datetime.datetime.strptime(ano + "-" + dia_ano_str + "-" + str(horario), "%Y-%j-%H")
                X.append({
                    "data": data,
                    "dia_ano": int(float(dia_ano)),           
                    "horario": int(float(horario)),     
                    "temp": float(row[3]),             
                    "hum" : float(row[4]),             
                    "pres": float(row[5]),             
                    "rad" : float(row[6]),        
                    "pres_delta": float(row[7]),
                    "pluv": float(row[8]),             
                    "choveu": int(float(row[9]))
                })
            if(linha > linha_max):
                break
            linha += 1

    dias_teste =200
    estacao = Estacao("Estação PROT", 0, 1200)
    treina = args.treina == "True"
    if(treina):
        print(treina_rnn(X, 24, 24, 24*dias_teste, estacao.id))
    else:
        print(previsao_rnn(X, 24, 24, 24*dias_teste, estacao.id))


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)