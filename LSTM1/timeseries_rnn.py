from statsmodels.tools.eval_measures import rmse
from sklearn.metrics                 import r2_score, mean_absolute_error as mae 
from tensorflow.keras.models         import model_from_json,  model_from_json
from tensorflow                      import keras
from tensorflow.keras                import layers
from tensorflow.keras.callbacks      import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing           import RobustScaler, StandardScaler
from sklearn.model_selection         import train_test_split
import os, gc, joblib, pandas as pd, numpy as np
from utils import *
from inicializacao import *
from matplotlib import pyplot as plt


# ======================== AUXILIARES =========================
EPSLON          = 0.0000001
RND_ST          = 142
I_TESTE_PADRAO  = 24
ARQ_NORMAL      = "normal"
ARQ_ENC_DEC     = "encoder_decoder"
ARQ_ENC_DEC_BID = "encoder_decoder_bidirecional"
BATCH_SIZE      = 2048
EPOCHS          = 1000
PATIENCE        = 100
STEPS = [24, 24]

# =================== CLASSE DE HIPERPARÂMETROS ===================
class MZDN_HP:
  def __init__(self, grandezas, error_f, h_layers=None, arq=None):
    self.grandezas              = grandezas # [0] contém X e [1] contém Y
    self.width_x, self.width_y  = len(self.grandezas[0]), len(self.grandezas[1])
    self.steps_b, self.steps_f  = STEPS     # [0] conterá Back e [1] conterá Forward
    self.error_f                = error_f
    self.h_layers               = h_layers
    self.arq                    = arq


# ============ CLASSE DE [PRE PROC + TREINO + PREV] ===========
class MZDN_HF:
  
  # CONSTRUTORES
  # Construtor 1: fornece apenas [diretorio] OU [modelo, scalers, diretorio] -> uso web!
  def __init__(self, diretorio, modelo=None, scalers=None, debug=True):
    
    self.diretorio = diretorio
    self.debug     = debug
    self.only_prev = True
    self.stats = []

    # Se infomou scalers usa. Senão -> busca no disco pelo dir
    if(scalers is not None):
      self.scalers_x = scalers[0]
      self.scalers_y = scalers[1]
    else:
      self.scalers_x = joblib.load(f'{diretorio}/scalers_x.gz')
      self.scalers_y = joblib.load(f'{diretorio}/scalers_y.gz')
      
    # Se infomou modelo usa. Senão -> busca no disco pelo dir
    if(modelo is not None):
      self.modelo    = modelo
    else:
      self.modelo    = self.carrega_modelo(diretorio)

    # Hiperparâmetros buscados do diretório
    hp_dict = np.load(f'{diretorio}/params.npy', allow_pickle='TRUE').item()
    self.hp = MZDN_HP(hp_dict["grandezas"], 
                     [hp_dict["steps_b"], hp_dict["steps_f"]],
                      hp_dict["error_f"],
                      hp_dict["h_layers"],
                      hp_dict["arq"])
    gc.collect()

  # Construtor 2: fornece apenas [diretorio, hiperparâmetros] -> scalers e modelo serão gerados -> uso lab!
  def __init__(self, diretorio, hp, debug=True):
    # Básico
    self.diretorio = diretorio
    self.nome      = diretorio.split("/")[-1]
    self.debug     = debug
    self.only_prev = False
    self.stats = []
    # Modelo e scalers serão gerados (construtor de treinamento)
    self.modelo    = None
    self.scalers_x = None
    self.scalers_y = None
    # Hiperparâmetros
    self.hp = hp

  def print_if_debug(self, args):
    if(self.debug):
      print(args)

  # Pré-processamento
  def gera_pre_proc_XY(self, _dict, iteracoes_teste, treinamento_e_salva_scalers):
    #################################################################### PRÉ PROCESSAMENTO ################################################################## 
    df = pd.DataFrame(_dict).set_index("data")
    X = self.__substitui_nulos_e_nan(df[self.hp.grandezas[0]])
    Y = self.__substitui_nulos_e_nan(df[self.hp.grandezas[1]])
    
    if(treinamento_e_salva_scalers):  
      self.scalers_y, self.scalers_x = StandardScaler(), StandardScaler()
      df_X = self.scalers_x.fit_transform(X) 
      df_Y = self.scalers_y.fit_transform(Y)

      joblib.dump(self.scalers_x, f'{self.diretorio}/scalers_x.gz')
      joblib.dump(self.scalers_y, f'{self.diretorio}/scalers_y.gz') 
      self.print_if_debug("\n SCALERS SALVOS NO DISCO!\n")  
    else:
      df_X = self.scalers_x.transform(X) 
      df_Y = self.scalers_y.transform(Y)   
  
    self.print_if_debug(pd.DataFrame(df_X).describe())
    self.print_if_debug(pd.DataFrame(df_Y).describe())

    janela_X, janela_Y = self.to_supervised(df_X, df_Y) 
    test_ratio = iteracoes_teste/len(janela_X)

    X_train, X_test = train_test_split(janela_X, test_size = test_ratio, shuffle = False, random_state = RND_ST) 
    Y_train, Y_test = train_test_split(janela_Y, test_size = test_ratio, shuffle = False, random_state = RND_ST) 
    return [df_X, df_Y], [X_train, Y_train], [X_test, Y_test]

  # Auxiliar no pré-processamento
  def __substitui_nulos_e_nan(self, df):
    for grandeza in df.columns:
      if(df[grandeza].dtypes != 'float'):
        if(self.debug):
          self.print_if_debug(f'{grandeza} não é float (ignorado).')
        continue
      else:
        media = float(df[grandeza].mean())
        if(self.debug):
          self.print_if_debug(f'{grandeza} é float ==> NaNs sobrescritos pela média ({media})!')
        df[grandeza] = df[grandeza].bfill().fillna(media)
    return df
  
  def carrega_e_compila(self, nome): 
    loaded_model = self.carrega_modelo(nome)
    loaded_model = self.compila(loaded_model)
    return loaded_model

  def carrega_modelo(nome): 
    # load json and create model
    json_file = open(f'{nome}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close() 
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"{nome}.h5")
    gc.collect()
    return loaded_model
  
  def compila(self, model): 
    model.compile(loss=self.hp.error_f, optimizer='nadam', metrics=[self.hp.error_f]) 
    return model
  
  def salva_modelo(self, model, nome): 
    if(self.only_prev):
      raise Exception("Esta é uma instância apenas de previsão, não é permitido: Retreinar; Ressalvar modelo/scalers.")
    
    # Salva modelo
    model_json = model.to_json()
    with open(f"{nome}.json", "w") as json_file:
        json_file.write(model_json) 
    model.save_weights(f"{nome}.h5") 

    # Salva hiperparâmetros
    hp_dict = {
      "grandezas" : self.hp.grandezas,
      "error_f"   : self.hp.error_f,
      "steps_f"   : self.hp.steps_f,
      "steps_b"   : self.hp.steps_b,
      "h_layers"  : self.hp.h_layers,
      "arq"       : self.hp.arq,
    }
    np.save(f"{self.diretorio}/params.npy", hp_dict)
    gc.collect()
    self.print_if_debug("Saved model to disk")

  def to_supervised(self, x_input, y_input):
    x, y         = [], []  
    #loop de dias => supervised com janela móvel de [self.hp.steps_b] passos traseiros 
    #                e [self.hp.steps_f] passos dianteiros
    for i in range(self.hp.steps_b, len(x_input) - self.hp.steps_f):   
      _x_aux, _y_aux = np.array(x_input[i - self.hp.steps_b : i]), np.array(y_input[i : i + self.hp.steps_f])
      if(len(_x_aux) > 0):
        x.append(_x_aux) 
        y.append(_y_aux)     

    return np.array(x), np.array(y)

  def evaluate_model(self, Y_true_arg, y_predicted_arg):
    scores_mae    = [] 
    scores_rmse   = []
    scores_smape  = []
    scores_ac     = [] 
    ac_cat        = [] 
    ac_pluv       = []
    r_2           = []
    for i in range(Y_true_arg.shape[1]):
      Y_true      =  [(float(y[i]) if y[i] is not None else float(EPSLON)) for y in Y_true_arg]
      y_predicted =  [(float(y[i]) if y[i] is not None else float(EPSLON)) for y in y_predicted_arg]  

      scores_mae.append( formata_2_casas(float( mae(Y_true, y_predicted))))
      scores_rmse.append(formata_2_casas(float(rmse(Y_true, y_predicted))))
      s = formata_2_casas(float(smape(Y_true, y_predicted)))
      scores_smape.append(float(s))
      scores_ac.append(formata_2_casas(float(100-s)))   
      ac_cat.append(None)
      ac_pluv.append(AcuraciaChuvaUtil.get_acuracia_distribuicao_diaria(Y_true, y_predicted, 0, 24, self.hp.steps_b, self.hp.steps_f) if i == 5 else None) 
      r_2.append(formata_2_casas(100*r2_score(Y_true, y_predicted)) if i != 5 else None)
      if(i==4):
        obj_testagem      = AcuraciaPluvUtil.get_acuracia_distribuicao_diaria(Y_true, y_predicted, 0, 24,  self.hp.steps_b,  self.hp.steps_f, True)  
        scores_mae[-1]    = formata_2_casas(obj_testagem["mae"])
        scores_rmse[-1]   = formata_2_casas(obj_testagem["rmse"])
        ac_pluv[-1]       = formata_2_casas(obj_testagem["ac_cat"]) 
        r_2[-1]           = formata_2_casas(obj_testagem["r2"]*100)
        s                 = formata_2_casas(obj_testagem["smape"])
        scores_smape[-1]  = s
        scores_ac[-1]     = formata_2_casas(100-s) 
    return [{
      "i"       : i,
      "mae"     : scores_mae[i],
      "rmse"    : scores_rmse[i],
      "smape"   : scores_smape[i],
      "ac"      : scores_ac[i],
      "ac_cat"  : ac_cat[i],
      "r_2"     : r_2[i],
      "ac_pluv" : ac_pluv[i],
      } for i in range(len(scores_mae))]
 
  def retorna_prev_e_erros(self, prev_test, Y_true, prev, funcao_analise=None):
    scores      = self.evaluate_model(Y_true, prev_test) 
    prev_finais = []
    intervalos  = []
    analises    = []
    prev_list   = prev[-self.hp.steps_f:].tolist()
    for i in range(prev.shape[1]):
      prev_el = [(float(el[i]) if el[i] is not None else float(0.001)) for el in prev_list]
      prev_finais.append(prev_el) 
      intervalos.append(MZDN_HF.intervalo_confianca_generico(prev_el, scores[i]["rmse"])) 
      if(funcao_analise is not None):
        analises.append(funcao_analise(prev_el))  
    return {
      "prev":       prev_finais,
      "intervalos": intervalos,
      "analise":    analises,
      "score":      scores
    }
        
  @staticmethod
  def intervalo_confianca_generico(prev, erro): 
    lista_intervalo = []
    propagacao_exp  = 1
    for i in range(len(prev)):
      erro*=propagacao_exp
      lista_intervalo.append({"ci_up": float(prev[i] + erro), "ci_down": float(prev[i]  - erro)})
       
    return lista_intervalo

  def __lstm_encoder_decoder_bidireccional(self):
    model = keras.Sequential() 
    # Encoder (bidirectional)
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(
      layers.LSTM(self.hp.h_layers, input_shape=(self.hp.steps_b, self.hp.width_x), dropout=0.5)
    ))
    # Enc -> Dec
    model.add(layers.RepeatVector(self.hp.steps_f))    
    # Decoder (unidirectional)
    model.add(layers.LSTM(self.hp.h_layers, return_sequences=True, dropout=0.5))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(self.hp.width_y)))   
    return model 
  
  def __lstm_encoder_decoder(self):
    model = keras.Sequential() 
    # Encoder (bidirectional)
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(self.hp.h_layers, input_shape=(self.hp.steps_b, self.hp.width_x), dropout=0.5))
    # Enc -> Dec
    model.add(layers.RepeatVector(self.hp.steps_f))    
    # Decoder (unidirectional)
    model.add(layers.LSTM(self.hp.h_layers, return_sequences=True, dropout=0.5))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(self.hp.width_y)))   
    return model 
  
  def __lstm_normal(self):
    model = keras.Sequential()
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(self.hp.h_layers, return_sequences=True, dropout=0.5))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(self.hp.width_y)))   
    return model 

  def __calcula_stats_e_salva(self, model, history, XY_train, XY_test, early_stopping_monitor):
    _, train_error_f_stat = model.evaluate(XY_train[0], XY_train[1], verbose=0)
    _, test_error_f_stat  = model.evaluate(XY_test[0], XY_test[1], verbose=0)
    val_error_f_stat = history.history[f'val_{self.hp.error_f}'][-1]
    stat_dict = {
      "nome": self.nome,
      "error_f": self.hp.error_f,
      "treino": train_error_f_stat,
      "validacao": val_error_f_stat,
      "teste": test_error_f_stat,
      "epoch_parada": early_stopping_monitor.stopped_epoch
    }
    self.stats.append(stat_dict)
    self.print_if_debug(stat_dict)

    # Salva estatísticas locais (última rede) e cumulativas (todas as redes da bateria)
    # Locais
    with open(f'{self.diretorio}/stats.csv', 'w') as f:
      w = csv.DictWriter(f, stat_dict.keys())
      w.writeheader()
      w.writerow(stat_dict)
    # Cumulativos
    with open(f'{self.diretorio}/stats.csv', 'w') as f:
      w = csv.DictWriter(f, stat_dict.keys())
      w.writeheader()
      for stat in self.stats:
        w.writerow(stat)
    # Salva gráficos de stats locais e cumulativos
    # Locais
    fg, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.plot(history.history[self.hp.error_f],          label=f'{self.nome}: train. {self.hp.error_f} ->')
    ax.plot(history.history[f'val_{self.hp.error_f}'], label=f'{self.nome}: valid. {self.hp.error_f} ->')
    ax.legend()
    fg.savefig(f"{self.diretorio}/metricas.pdf", bbox_inches='tight')
    fg.savefig(f"{self.diretorio}/metricas.png", bbox_inches='tight')
    # Cumulativos
    plt.plot(history.history[self.hp.error_f],          label=f'{self.nome}: train. {self.hp.error_f}')
    plt.plot(history.history[f'val_{self.hp.error_f}'], label=f'{self.nome}: valid. {self.hp.error_f}')
    plt.legend()
    plt.savefig(f"{self.diretorio}/metricas_cumulativas.pdf", bbox_inches='tight')

  def treinar(self, dados_form, iteracoes_teste=I_TESTE_PADRAO):     
    if(self.only_prev):
      raise Exception("Esta é uma instância apenas de previsão, não é permitido: Retreinar; Ressalvar modelo/scalers.")
    
    XY, XY_train, XY_test = self.gera_pre_proc_XY(dados_form, iteracoes_teste, True) 
    # ##################################################################### LSTM MODEL
    early_stopping_monitor = EarlyStopping(monitor=self.hp.error_f, patience=PATIENCE, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath = f"{self.diretorio}/checkpoint", verbose=0, save_best_only=True)

    if(self.hp.arq == ARQ_NORMAL):
      model = self.__lstm_normal()
    elif(self.hp.arq == ARQ_ENC_DEC):
      model = self.__lstm_encoder_decoder()
    elif(self.hp.arq == ARQ_ENC_DEC_BID):
      model = self.__lstm_encoder_decoder_bidireccional()
    else:
      raise Exception(f"Uma arquitetura desconhecida foi solicitada. Esperava-se [\"{ARQ_NORMAL}\", \"{ARQ_ENC_DEC}\", \"{ARQ_ENC_DEC_BID}\"] -> recebido: \"{self.hp.arq}\"")

    model = self.compila(model)
    history = model.fit(
      XY_train[0],            # X
      XY_train[1],            # Y
      validation_split = 0.2,
      batch_size = BATCH_SIZE,
      epochs = EPOCHS, 
      shuffle = False,  
      callbacks=[early_stopping_monitor, checkpointer], 
      verbose=2,
    )
  
    # Salva o modelo
    self.salva_modelo(model, self.diretorio)
    
    # Gera relatórios estatísticos do treinamento
    self.__calcula_stats_e_salva(model, history, XY_train, XY_train, early_stopping_monitor)
    
  def prever(self, dados_form, iteracoes_teste=I_TESTE_PADRAO, inclui_compostas=None, compostas_args=None):     
    
    df_XY, _, test_XY = self.gera_pre_proc_XY(dados_form, iteracoes_teste) 
    
    self.print_if_debug(f"Modelo carregado do disco \n SUMÁRIO DE MODELO: {self.modelo.summary()}\n X_test shape = {test_XY[0].shape}")

    #--------------- prepara prev ---------------
    base_prev_x = np.array([df_XY[0][-self.hp.steps_b:,:]])

    score = self.modelo.evaluate(test_XY[0], test_XY[1], verbose= 0 if self.debug else 1)

    #------------------- debug ------------------
    self.print_if_debug(f"{self.modelo.metrics_names[1]}: {score[1]}" )
    self.print_if_debug(f"ÚLTIMAS 24 USADAS S/ INVERSE SCALING: \n {[el for el in base_prev_x[:, :, :self.hp.width_x]]}\n")
    self.print_if_debug(f"ÚLTIMAS 24 USADAS C/ INVERSE SCALING: \n {[self.scalers_x.inverse_transform(el) for el in base_prev_x[:, :, :self.hp.width_x]]}\n")  

    df_pred       = self.modelo.predict(base_prev_x[-1:])
    prev          = np.array([self.scalers_y.inverse_transform(el) for el in df_pred]).reshape(-1, self.hp.width_y)
    
    #--------------- prepara teste --------------- 
    df_pred_test  = self.modelo.predict(test_XY[0]) 
    prev_test     = np.array([self.scalers_y.inverse_transform(el) for el in df_pred_test]).reshape(-1, self.hp.width_y)
    Y_true_test   = np.array(self.scalers_y.inverse_transform(test_XY[1].reshape(-1, self.hp.width_y)))
  
    #--------------- pos proc ---------------
    if(inclui_compostas is not None):
      prev          = inclui_compostas(prev,        compostas_args)
      prev_test     = inclui_compostas(prev_test,   compostas_args)
      Y_true_test   = inclui_compostas(Y_true_test, compostas_args) 
    
    return self.retorna_prev_e_erros(prev_test, Y_true_test, prev)
    