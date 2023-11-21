import math, csv, datetime, argparse
from timeseries_rnn import *
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--rank_models',                default=False)
    parser.add_argument('--stat_DOUT',                  default=False)
    parser.add_argument('--stat_HL',                    default=False)
    parser.add_argument('--stat_BCW',                   default=False)
    parser.add_argument('--stat_BID',                   default=False)
    parser.add_argument('--treina',                     default=True)
    parser.add_argument('--grandezas',                  default="X_Ysp")
    parser.add_argument('--efunc',                      default="mae")
    parser.add_argument('--linha',                      default=0)
    parser.add_argument('--linha_max',                  default=None)
    parser.add_argument('--arquivo',                    default='clima_bsb.csv')
    # 1 ano (dados horários) => Último [2014, 2020] na base clima_bsb => [2020] p/ teste
    parser.add_argument('--iteracoes_teste',            default=365*24)
    parser.add_argument('--batch_sizes',                default=None)
    return parser

def printa_grafico_grupos(grupo_key, filter_hl=None):
    grupos = MZDN_HF.rank_models("modelos", grupo_key, filter_hl)
    mae, param = [], []
    for grupo in grupos:
        param.append(grupo[0]["prop_grupo"])
        soma, count = float(0), float(0)
        for el in grupo:
            soma += float(el["erro_teste"])
            count += 1
        mae.append(soma/count)
    return mae, param

def main(args):
    
    marker = itertools.cycle(('o', '*', 'v', 's', 'p', '1'))
    if(args.rank_models == "True"):
        # print tabela
        print("Gerando ranking de modelos!")
        grupos = MZDN_HF.rank_models("modelos", "erro_f")
        for grupo in grupos:
            for el in grupo:
                print(el)
        return
    elif(args.stat_DOUT == "True"):
        print("Gerando gráfico MAE x Dropout (DOUT)")
        fg, ax = plt.subplots()
        hls = [10, 50, 100, 150]
        maes_0, maes_025, maes_050 = [], [], []
        for hl in hls:
            mae, _ = printa_grafico_grupos("dropout", hl)
            maes_0.append(mae[0])
            maes_025.append(mae[1])
            maes_050.append(mae[2])

        ax.plot(np.array(hls), np.array(maes_0),   marker=next(marker), label="0")
        ax.plot(np.array(hls), np.array(maes_025), marker=next(marker), label="0.25")
        ax.plot(np.array(hls), np.array(maes_050), marker=next(marker), label="0.5")
        ax.legend()
        ax.xaxis.set_ticks(hls)
        ax.set_title("MAE x HL por dropout (DOUT)")
        fg.savefig("relatorios_gerais/dropout.pdf")
        fg.savefig("relatorios_gerais/dropout.png")
        plt.show()
        return
    elif(args.stat_HL == "True"):
        print("Gerando gráfico MAE x Hidden Layers (HL)")
        fg, ax = plt.subplots()
        mae, param = printa_grafico_grupos("hidden_layers")
        ax.plot(np.array(param), np.array(mae), marker=next(marker), label="geral")
        ax.legend()
        ax.xaxis.set_ticks([10, 50, 100, 150])
        ax.set_title("MAE x Hidden Layers (HL)")
        fg.savefig("relatorios_gerais/hidden_layers.pdf")
        fg.savefig("relatorios_gerais/hidden_layers.png")
        plt.show()
        return
    elif(args.stat_BCW == "True"):
        print("Gerando gráfico MAE x Janelamento traseiro (BID)")
        fg, ax = plt.subplots()
        hls = [10, 50, 100, 150]
        maes_24, maes_48 = [], []
        for hl in hls:
            mae, _ = printa_grafico_grupos("back_window", hl)
            maes_24.append(mae[0])
            maes_48.append(mae[1])

        ax.plot(np.array(hls), np.array(maes_24), marker=next(marker), label="24")
        ax.plot(np.array(hls), np.array(maes_48), marker=next(marker), label="48")
        ax.legend()
        ax.xaxis.set_ticks(hls)
        ax.set_title("MAE x HL por Janelamento traseiro (BCW)")
        fg.savefig("relatorios_gerais/back_window.pdf")
        fg.savefig("relatorios_gerais/back_window.png")
        plt.show()
        return
    elif(args.stat_BID == "True"):
        print("Gerando gráfico MAE x Bidirecionalidade (BID)")
        fg, ax = plt.subplots()
        hls = [10, 50, 100, 150]
        maes_uni, maes_bid = [], []
        for hl in hls:
            mae, _ = printa_grafico_grupos("arq", hl)
            maes_uni.append(mae[0])
            maes_bid.append(mae[1])

        ax.plot(np.array(hls), np.array(maes_uni), marker=next(marker), label="Unidirecional")
        ax.plot(np.array(hls), np.array(maes_bid), marker=next(marker), label="Bidirecional")
        ax.legend()
        ax.xaxis.set_ticks(hls)
        ax.set_title("MAE x HL por Bidirecionalidade (BID)")
        fg.savefig("relatorios_gerais/arq.pdf")
        fg.savefig("relatorios_gerais/arq.png")
        plt.show()
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
            if(linha_max is not None and linha > int(linha_max)):
                break
            linha += 1

    grandezas_list  = {
        "X_Y" : [
                ["temp", "hum", "pres", "rad", "pluv"],      
                ["temp", "hum", "pres", "rad", "pluv"]       
            ],
        "X_Ysp" : [
                ["temp", "hum", "pres", "rad", "pluv"],      
                ["temp", "hum", "pres", "rad"]       
            ],
    }
    
    # Filtra apenas as grandezas desejadas
    grandezas = grandezas_list[args.grandezas]
    
    
    # Com referência ao paper anterior, poram subtraídas as grandezas Pres e Chuva (categ). Pres pois foi substituída pelo
    # seu delta horário a fim de resolver incongruências de nível e Chuva (Categ) pois não se pode ter valores discretos na
    # saída com uma rede com função de erro MSE/MAE. Precisa-se de uma rede dedicada para esta grandeza.
    
    hps = [
        #----------------------------------------------------------------------------------
        #---------------------------------- bidirecional ----------------------------------
        # -----  batch size 32
        # -----  dropout 0
        MZDN_HP(grandezas, args.efunc, 10,  48, ARQ_ENC_DEC_BID, 0),
        MZDN_HP(grandezas, args.efunc, 10,  24, ARQ_ENC_DEC_BID, 0),
        MZDN_HP(grandezas, args.efunc, 50,  48, ARQ_ENC_DEC_BID, 0),
        MZDN_HP(grandezas, args.efunc, 50,  24, ARQ_ENC_DEC_BID, 0),
        MZDN_HP(grandezas, args.efunc, 100, 48, ARQ_ENC_DEC_BID, 0),
        MZDN_HP(grandezas, args.efunc, 100, 24, ARQ_ENC_DEC_BID, 0),
        MZDN_HP(grandezas, args.efunc, 150, 48, ARQ_ENC_DEC_BID, 0),
        MZDN_HP(grandezas, args.efunc, 150, 24, ARQ_ENC_DEC_BID, 0),
        # -----  dropout 0.25
        MZDN_HP(grandezas, args.efunc, 10,  48, ARQ_ENC_DEC_BID, 0.25),
        MZDN_HP(grandezas, args.efunc, 10,  24, ARQ_ENC_DEC_BID, 0.25),
        MZDN_HP(grandezas, args.efunc, 50,  48, ARQ_ENC_DEC_BID, 0.25),
        MZDN_HP(grandezas, args.efunc, 50,  24, ARQ_ENC_DEC_BID, 0.25),
        MZDN_HP(grandezas, args.efunc, 100, 48, ARQ_ENC_DEC_BID, 0.25),
        MZDN_HP(grandezas, args.efunc, 100, 24, ARQ_ENC_DEC_BID, 0.25),
        MZDN_HP(grandezas, args.efunc, 150, 48, ARQ_ENC_DEC_BID, 0.25),
        MZDN_HP(grandezas, args.efunc, 150, 24, ARQ_ENC_DEC_BID, 0.25),
        # -----  dropout 0.50
        MZDN_HP(grandezas, args.efunc, 10,  48, ARQ_ENC_DEC_BID, 0.5),
        MZDN_HP(grandezas, args.efunc, 10,  24, ARQ_ENC_DEC_BID, 0.5),
        MZDN_HP(grandezas, args.efunc, 50,  48, ARQ_ENC_DEC_BID, 0.5),
        MZDN_HP(grandezas, args.efunc, 50,  24, ARQ_ENC_DEC_BID, 0.5),
        MZDN_HP(grandezas, args.efunc, 100, 48, ARQ_ENC_DEC_BID, 0.5),
        MZDN_HP(grandezas, args.efunc, 100, 24, ARQ_ENC_DEC_BID, 0.5),
        MZDN_HP(grandezas, args.efunc, 150, 48, ARQ_ENC_DEC_BID, 0.5),
        MZDN_HP(grandezas, args.efunc, 150, 24, ARQ_ENC_DEC_BID, 0.5),

        
        #-----------------------------------------------------------------------------------
        #---------------------------------- unidirecional ----------------------------------
        # -----  batch size 32
        # -----  dropout 0
        MZDN_HP(grandezas, args.efunc, 10,  48, ARQ_ENC_DEC, 0),
        MZDN_HP(grandezas, args.efunc, 10,  24, ARQ_ENC_DEC, 0),
        MZDN_HP(grandezas, args.efunc, 50,  48, ARQ_ENC_DEC, 0),
        MZDN_HP(grandezas, args.efunc, 50,  24, ARQ_ENC_DEC, 0),
        MZDN_HP(grandezas, args.efunc, 100, 48, ARQ_ENC_DEC, 0),
        MZDN_HP(grandezas, args.efunc, 100, 24, ARQ_ENC_DEC, 0),
        MZDN_HP(grandezas, args.efunc, 150, 48, ARQ_ENC_DEC, 0),
        MZDN_HP(grandezas, args.efunc, 150, 24, ARQ_ENC_DEC, 0),
        # -----  dropout 0.25
        MZDN_HP(grandezas, args.efunc, 10,  48, ARQ_ENC_DEC, 0.25),
        MZDN_HP(grandezas, args.efunc, 10,  24, ARQ_ENC_DEC, 0.25),
        MZDN_HP(grandezas, args.efunc, 50,  48, ARQ_ENC_DEC, 0.25),
        MZDN_HP(grandezas, args.efunc, 50,  24, ARQ_ENC_DEC, 0.25),
        MZDN_HP(grandezas, args.efunc, 100, 48, ARQ_ENC_DEC, 0.25),
        MZDN_HP(grandezas, args.efunc, 100, 24, ARQ_ENC_DEC, 0.25),
        MZDN_HP(grandezas, args.efunc, 150, 48, ARQ_ENC_DEC, 0.25),
        MZDN_HP(grandezas, args.efunc, 150, 24, ARQ_ENC_DEC, 0.25),
        # -----  dropout 0.50
        MZDN_HP(grandezas, args.efunc, 10,  48, ARQ_ENC_DEC, 0.5),
        MZDN_HP(grandezas, args.efunc, 10,  24, ARQ_ENC_DEC, 0.5),
        MZDN_HP(grandezas, args.efunc, 50,  48, ARQ_ENC_DEC, 0.5),
        MZDN_HP(grandezas, args.efunc, 50,  24, ARQ_ENC_DEC, 0.5),
        MZDN_HP(grandezas, args.efunc, 100, 48, ARQ_ENC_DEC, 0.5),
        MZDN_HP(grandezas, args.efunc, 100, 24, ARQ_ENC_DEC, 0.5),
        MZDN_HP(grandezas, args.efunc, 150, 48, ARQ_ENC_DEC, 0.5),
        MZDN_HP(grandezas, args.efunc, 150, 24, ARQ_ENC_DEC, 0.5),

    ]

    if(args.treina):
        for i in range(len(hps)):
            hp          = hps[i]
            diretorio   = f"modelos/{args.grandezas}/{args.efunc.upper()}_{hp.arq}_BAT{hp.batch_size}_DOUT{hp.dropout}_HL{hp.h_layers}_BCKW{hp.steps_b}"
            mzdn        = MZDN_HF(diretorio, hp, True)
            mzdn.treinar(X, args.iteracoes_teste)
    else:
        mzdn = MZDN_HF(args.diretorio)
        mzdn.prever(X, 0)



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)