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
    X = []
    with open('clima_bsb.csv') as _csv:
        _r = csv.reader(_csv, delimiter=',')
        linha= 0
        linha_max = 365*24*10
        iteracoes_teste = 30*24

        for row in _r:
            if linha > 0:
                horario = int(float(row[2]))
                data = datetime.datetime.strptime(row[0].split(" ")[0], "%Y-%m-%d")
                # considera rad apenas entre 8 e 16     (nossas estações causam deformidade às 6~7 e 17~18h)
                _rad = row[6] if horario>=8 and horario<=16 else 0
                vazio = math.isnan(float(row[3])) or float(row[3]) is None
                if(vazio):
                    continue
                X.append({
                    "data":     data,         
                    "horario":  int(row[1]), 
                    "dia_ano":  int(row[2]),     
                    "ano":      int(data.year),      
                    "temp":     float(row[3]) ,             
                    "hum" :     float(row[4]) ,             
                    "pres":     float(row[5]) ,             
                    "rad" :     float(_rad)   ,   
                    "pluv":     float(row[7]) ,    
                    "choveu":   int(row[8])   ,
                    "vel":      float(row[9]) ,
                    "dir":      float(row[10]),
                    "temp_d":   float(row[11]),       
                    "hum_d":    float(row[12]),       
                    "pres_d":   float(row[13]),         
                })
            if(linha > linha_max):
                break
            linha += 1

    treina = args.treina == "True"

    # Serão treinados cada um dos hps casos listados p/ cada arquitetura
    arquiteturas = [ARQ_NORMAL, ARQ_ENC_DEC, ARQ_ENC_DEC_BID]
    hps = [
        # ----------------- Varia hidden_layers
        MZDN_HP([ ["temp", "hum", "pres", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres", "rad", "pluv"]],      [24, 24], "mse", 100),

        MZDN_HP([ ["temp", "hum", "pres", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres", "rad", "pluv"]],      [24, 24], "mse", 200),
                
        MZDN_HP([ ["temp", "hum", "pres", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres", "rad", "pluv"]],      [24, 24], "mse", 400),

        # ----------------- Varia hidden_layers com pres_d
        MZDN_HP([ ["temp", "hum", "pres_d", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres_d", "rad", "pluv"]],    [24, 24], "mse", 100),
                
        MZDN_HP([ ["temp", "hum", "pres_d", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres_d", "rad", "pluv"]],    [24, 24], "mse", 200),
                
        MZDN_HP([ ["temp", "hum", "pres_d", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres_d", "rad", "pluv"]],    [24, 24], "mse", 400),

        # ----------------- Varia função de erro (fixa em máx h_layers, muda pres_d)
        MZDN_HP([ ["temp", "hum", "pres", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres", "rad", "pluv"]],      [24, 24], "mae", 400),

        MZDN_HP([ ["temp", "hum", "pres_d", "rad", "pluv", "ano", "dia_ano", "horario"],
                  ["temp", "hum", "pres_d", "rad", "pluv"]],    [24, 24], "mae", 400)
    ]
    diretorios = [f"__modelos/modelo_{i+1}" for i in range(len(hps))]
    if(treina):
        for arq in arquiteturas:
            for i, diretorio in enumerate(diretorios):
                hp = hps[i]
                hp.arq = arq
                mzdn = MZDN_HF(diretorio, hp, True)
                print(mzdn.treinar(X, iteracoes_teste))
    else:
        mzdn = MZDN_HF(args.diretorio)
        mzdn.prever(X)



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)