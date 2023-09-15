import math, os, gc, sys, joblib, pandas as pd, csv, numpy as np, tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from timeseries_rnn import *
from estacao import *
import datetime
import argparse
import gc, tensorflow as tf


def __keras_init():
    print( f"num gpus available {len(tf.config.experimental.list_physical_devices('gpu'))}" )
    # #################################################################### HARDWARE CONFIG
    GPU = False
    CPU = True
    NUM_CORES = 4
    if GPU:
        num_GPU = 1
        num_CPU = 2
    if CPU:
        num_CPU = 2
        num_GPU = 0
    _config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = NUM_CORES,
                                       inter_op_parallelism_threads = NUM_CORES, 
                                       allow_soft_placement         = True,
                                       device_count                 = {'CPU' : num_CPU, 'GPU' : num_GPU})
    K.set_session(tf.compat.v1.Session(config = _config))
    physical_devices = tf.config.list_physical_devices('GPU')
    if(len(physical_devices) > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    gc.collect()
    return None

__keras_init()


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--treina',                     default=True)
    parser.add_argument('--linha',                      default=0)
    parser.add_argument('--linha_max',                  default=None)
    parser.add_argument('--arquivo',                    default='clima_bsb.csv')
    # 1 ano (dados horários) => Último [2014, 2020] na base clima_bsb => [2020] p/ teste
    parser.add_argument('--iteracoes_teste',            default=365*24)
    return parser

def main(args):
    X = []
    with open(args.arquivo) as _csv:
        _r = csv.reader(_csv, delimiter=',')
        linha = args.linha
        linha_max = args.linha_max
        treina = args.treina
        iteracoes_teste = args.iteracoes_teste

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
            if(linha_max is not None and linha > linha_max):
                break
            linha += 1

    grandezas    = [ 
        ["temp", "hum", "pres_d", "rad", "pluv", "ano", "dia_ano", "horario"],  # X
        ["temp", "hum", "pres_d", "rad", "pluv"]                                # Y
    ]
    # Justificativa das grandezas e porque foram fixadas.
    #   - Temp, Hum, Rad, Pluv [X e Y]: Usadas no modelo do paper anterior. Necessário no p/ se ter histórico destas mesmas
    #                                   grandezas que queremos, também, prever. São contínuas e complexamente correlacionadas.
    #
    #   - Pres_d               [X e Y]: Delta horário da pressão. Evita transformações de altitude para compensar valores de 
    #                                   pressão nas estações de treino e teste. Erro na transformação de altitude (por imprecisão
    #                                   topográfica sobre o local o quaisquer outros motivos) e pequenas incongruências podem
    #                                   facilmente prejudicar a performance da rede. A análise de variações resolve este problema
    #                                   de nível.
    #
    #   - Ano, Dia             [X]:     Utilizado apenas em X para se ter uma ideia de quando foram registrados. Não
    #                                   importa apenas a sequência, mas o contexto sazonal dela: qual era a estação/mês?
    #                                   De qual ano? Ano importa devido ao El-Niño e El-Niña.
    #
    #   - Horário              [X]:     Poder-se-ia argumentar que apenas com Rad já se teria uma noção de horário, no entanto,
    #                                   dias mais ou menos nublados pela manhã/tarde podem afetar as leituras de início e fim
    #                                   para Rad e alterar a percepção de dia/noite. Para Rad=0, pode tanto se estar muito
    #                                   nublado como estar de noite (o sensor não é muito preciso em baixas irradiâncias). Faz-se
    #                                   mister, portanto, uma variável em [X] que auxilie na indicação horária.
    #
    # Com referência ao paper anterior, poram subtraídas as grandezas Pres e Chuva (categ). Pres pois foi substituída pelo
    # seu delta horário a fim de resolver incongruências de nível e Chuva (Categ) pois não se pode ter valores discretos na
    # saída com uma rede com função de erro MSE/MAE. Precisa-se de uma rede dedicada para esta grandeza.
    
    # Serão treinados cada um dos hps casos listados em HPs p/ cada arquitetura abaixo
    arquiteturas = [
        ARQ_ENC_DEC, 
        ARQ_ENC_DEC_BID
    ]
    hps = [
        MZDN_HP(grandezas, "mse", 100, 24),
        MZDN_HP(grandezas, "mse", 100, 48),
        MZDN_HP(grandezas, "mse", 100, 72),

        MZDN_HP(grandezas, "mse", 200, 24),
        MZDN_HP(grandezas, "mse", 200, 48),
        MZDN_HP(grandezas, "mse", 200, 72),

        MZDN_HP(grandezas, "mse", 400, 24),
        MZDN_HP(grandezas, "mse", 400, 48),
        MZDN_HP(grandezas, "mse", 400, 72),
    ]

    if(treina):
        for arq in arquiteturas:
            for i in range(len(hps)):
                hp = hps[i]
                hp.arq = arq
                diretorio = f"__modelos/{arq}/{hp.error_f}_{hp.h_layers}HL_{hp.steps_b}B"
                mzdn = MZDN_HF(diretorio, hp, True)
                print(mzdn.treinar(X, iteracoes_teste))
    else:
        mzdn = MZDN_HF(args.diretorio)
        mzdn.prever(X)



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)