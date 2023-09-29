import math, csv, datetime, argparse
from timeseries_rnn import *

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--rank_models',                default=False)
    parser.add_argument('--treina',                     default=True)
    parser.add_argument('--grandezas',                  default="ApDX_ApDY")
    parser.add_argument('--efunc',                      default="mse")
    parser.add_argument('--linha',                      default=0)
    parser.add_argument('--linha_max',                  default=None)
    parser.add_argument('--arquivo',                    default='clima_bsb.csv')
    # 1 ano (dados horários) => Último [2014, 2020] na base clima_bsb => [2020] p/ teste
    parser.add_argument('--iteracoes_teste',            default=365*24)
    parser.add_argument('--batch_sizes',                default=None)
    return parser

def main(args):

    if(args.rank_models == "True"):
        print("Gerando ranking de modelos!")
        print([str(x)+"\n" for x in MZDN_HF.rank_models("modelos")])
        return

    X = []
    with open(args.arquivo) as _csv:
        _r = csv.reader(_csv, delimiter=',')
        linha           = args.linha
        linha_max       = args.linha_max

        for row in _r:
            if linha > 0:
                horario = int(float(row[1]))
                data = datetime.datetime.strptime(row[0].split(" ")[0], "%Y-%m-%d")
                # considera rad apenas entre 8 e 16     (nossas estações causam deformidade às 6~7 e 17~18h)
                _rad = row[6] if horario >= 8 and horario <= 16 else 0
                vazio = math.isnan(float(row[3])) or float(row[3]) is None
                if(vazio):
                    continue
                X.append({
                    "data":     data,         
                    "horario":  horario, 
                    "dia_ano":  int(row[2]),     
                    "ano":      int(data.year),      
                    "temp":     float(row[3]),             
                    "hum" :     float(row[4]),             
                    "pres":     float(row[5]),             
                    "rad" :     float(_rad),   
                    "pluv":     float(row[7]),    
                    "choveu":   int(row[8]),
                    "vel":      float(row[9]),
                    "dir":      float(row[10]),
                    "temp_d":   float(row[11]),       
                    "hum_d":    float(row[12]),       
                    "pres_d":   float(row[13]),         
                })
            if(linha_max is not None and linha > linha_max):
                break
            linha += 1

    grandezas_list  = {
        # X: Abs + del      Y: Abs
        "ADX_AY" : [
                ["temp", "hum", "pres","temp_d", "hum_d", "pres_d", "rad", "pluv", "ano", "dia_ano", "horario"],  # X
                ["temp", "hum", "pres", "rad", "pluv"]                                                            # Y
            ],
        # X: Abs            Y: Abs
        "AX_AY" : [
                ["temp", "hum", "pres", "rad", "pluv", "ano", "dia_ano", "horario"],                              # X
                ["temp", "hum", "pres", "rad", "pluv"]                                                            # Y
            ],
        # X: Abs + pres_d   Y: Abs + pres_d
        "ApDX_ApDY" : [
                ["temp", "hum", "pres_d", "rad", "pluv", "ano", "dia_ano", "horario"],                            # X
                ["temp", "hum", "pres_d", "rad", "pluv"]                                                          # Y
            ],
    }
    
    # Filtra apenas as grandezas desejadas
    grandezas = grandezas_list[args.grandezas]
    
    
    # Com referência ao paper anterior, poram subtraídas as grandezas Pres e Chuva (categ). Pres pois foi substituída pelo
    # seu delta horário a fim de resolver incongruências de nível e Chuva (Categ) pois não se pode ter valores discretos na
    # saída com uma rede com função de erro MSE/MAE. Precisa-se de uma rede dedicada para esta grandeza.
    
    # Serão treinados cada um dos hps casos listados em HPs p/ cada arquitetura abaixo
    arquiteturas = [ ARQ_ENC_DEC_BID, ARQ_ENC_DEC ]
    hps = [
        MZDN_HP(grandezas, args.efunc, 600, 72),
        MZDN_HP(grandezas, args.efunc, 600, 48),
        MZDN_HP(grandezas, args.efunc, 600, 24),

        MZDN_HP(grandezas, args.efunc, 200, 72),
        MZDN_HP(grandezas, args.efunc, 200, 48),
        MZDN_HP(grandezas, args.efunc, 200, 24),

        MZDN_HP(grandezas, args.efunc, 400, 72),
        MZDN_HP(grandezas, args.efunc, 400, 48),
        MZDN_HP(grandezas, args.efunc, 400, 24),
    ]
    
    batch_sizes = [1024, 512, 256]
    if(args.batch_sizes is not None):
        batch_sizes = args.batch_sizes.split(',')
        batch_sizes = [int(i) for i in batch_sizes]

    if(args.treina):
        for batch_size in batch_sizes:
            for arq in arquiteturas:
                for i in range(len(hps)):
                    hp          = hps[i]
                    hp.arq      = arq
                    diretorio   = f"modelos/{args.grandezas}/{args.efunc}_{arq}_BT{batch_size}_HL{hp.h_layers}_BS{hp.steps_b}"
                    mzdn        = MZDN_HF(diretorio, hp, True, batch_size)
                    mzdn.treinar(X, args.iteracoes_teste)
    else:
        mzdn = MZDN_HF(args.diretorio)
        mzdn.prever(X, 0)



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)