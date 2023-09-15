#region IMPORTS
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics                 import r2_score, mean_absolute_error as mae 
from tensorflow.keras.models         import *
from tensorflow                      import keras
from tensorflow.keras                import layers
from tensorflow.keras.callbacks      import EarlyStopping, ModelCheckpoint
from sklearn.metrics                 import  mean_absolute_error as mae  
from sklearn.preprocessing           import RobustScaler, StandardScaler
from sklearn.model_selection         import train_test_split
import os, gc, joblib, csv, pandas as pd, numpy as np
from matplotlib import pyplot as plt
#endregion


#region ======================= AUXILIARES ==========================
EPSLON          = 0.0000001
RND_ST          = 142
I_TESTE_PADRAO  = 24
ARQ_ENC_DEC     = "encoder_decoder"
ARQ_ENC_DEC_BID = "encoder_decoder_bidirectional"
BATCH_SIZE      = 2048
EPOCHS          = 200
PATIENCE        = 25

def __formata_2_casas(num): 
  return float('{0:.2f}'.format(num)) if num is not None else None

def __smape(A, F):
  A       = np.array(A)
  F       = np.array(F)
  epsilon = 0.1
  resp    = 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + epsilon))
  return resp if resp == resp else 0 #se não NaN
#endregion


#region ================ CLASSE DE HIPERPARÂMETROS ==================
class MZDN_HP:
  def __init__(self, grandezas, error_f, h_layers, steps, arq=None):
    self.grandezas              = grandezas # [0] contém X e [1] contém Y
    self.width_x, self.width_y  = len(self.grandezas[0]), len(self.grandezas[1])
    self.steps_b, self.steps_f  = steps, 24
    self.error_f                = error_f
    self.h_layers               = h_layers
    self.arq                    = arq

  # Persiste os hiperparâmetros no diretório especificado
  def salvar(self, diretorio):
    hp_dict = {
      "grandezas" : self.grandezas,
      "error_f"   : self.error_f,
      "steps_f"   : self.steps_f,
      "steps_b"   : self.steps_b,
      "h_layers"  : self.h_layers,
      "arq"       : self.arq,
    }
    with open(f'{diretorio}/params.csv', 'w') as f:
      w = csv.DictWriter(f, hp_dict.keys())
      w.writeheader()
      w.writerow(hp_dict)
    np.save(f"{diretorio}/params.npy", hp_dict)
    gc.collect()
#endregion


#region ============ CLASSE DE [PRE PROC + TREINO + PREV] ===========
class MZDN_HF:
  
  def __init__(self, diretorio, hp=None, debug=True):
    
    self.diretorio = diretorio
    self.nome      = diretorio.split("__modelos")[-1]
    self.debug     = debug
    self.stats     = []
    self.checkpointed_model_path = f"{diretorio}/checkpointed_model"
    self.scalers_x_path          = f'{diretorio}/scalers/scalers_x.gz'
    self.scalers_y_path          = f'{diretorio}/scalers/scalers_y.gz'
    self.hp_dict_path            = f'{diretorio}/modelo/params.npy'
    self.stat_csv_path           = f'{diretorio}/relatorio/relatorio.csv'
    self.stat_pdf_path           = f'{diretorio}/relatorio/relatorio.pdf'
    self.stat_png_path           = f'{diretorio}/relatorio/relatorio.png'

    if(hp is not None):
      # Se forneceu hp, é uma instância de treinamento (uso lab, apenas)
      self.only_prev = False
      # Hiperparâmetros
      self.hp        = hp
      self.hp.salvar(self.diretorio)
      # Modelo e scalers serão gerados
      self.scalers_x = None
      self.scalers_y = None
      self.modelo    = None
    else:
      # Se não forneceu hp, é uma instância de previsão (uso web -> produção)
      self.only_prev = True
      # Hiperparâmetros buscados do diretório
      hp_dict = np.load(self.hp_dict_path, allow_pickle='TRUE').item()
      self.hp = MZDN_HP(hp_dict["grandezas"], 
                      [hp_dict["steps_b"], hp_dict["steps_f"]],
                        hp_dict["error_f"],
                        hp_dict["h_layers"],
                        hp_dict["arq"])
      # Recupera scalers e model do diretorio 
      self.scalers_x = joblib.load(self.scalers_x_path)
      self.scalers_y = joblib.load(self.scalers_y_path)
      self.modelo    = self.__get_arquitetura_compilada()
      self.modelo    = tf.keras.models.load_model(self.checkpointed_model_path)

  #region AUXILIARES
  def print_if_debug(self, args):
    if(self.debug):
      print(args)
  #endregion

  #region PRÉ-PROCESSAMENTO
  def gera_pre_proc_XY(self, _dict, iteracoes_teste, treinamento_e_salva_scalers):
    #################################################################### PRÉ PROCESSAMENTO ################################################################## 
    df = pd.DataFrame(_dict).set_index("data")
    X = self.__substitui_nulos_e_nan(df[self.hp.grandezas[0]])
    Y = self.__substitui_nulos_e_nan(df[self.hp.grandezas[1]])
    
    if(treinamento_e_salva_scalers):  
      self.scalers_y, self.scalers_x = StandardScaler(), StandardScaler()
      df_X = self.scalers_x.fit_transform(X) 
      df_Y = self.scalers_y.fit_transform(Y)

      joblib.dump(self.scalers_x, self.scalers_x_path)
      joblib.dump(self.scalers_y, self.scalers_y_path) 
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
  #endregion
  
  #region ARQUITETURAS LSTM
  # encoder-decoder necessário pois se trata de um seq2seq com tamanhos assimétricos. Uma LSTM simples não funcionaria.
  def __lstm_encoder_decoder_bidireccional(self):
    model = keras.Sequential() 
    # Encoder (bidirectional)
    model.add(layers.Dropout(0.8))
    model.add(layers.Bidirectional(
      layers.LSTM(self.hp.h_layers, input_shape=(self.hp.steps_b, self.hp.width_x), dropout=0.5)
    ))
    model.add(layers.RepeatVector(self.hp.steps_f))    
    # Decoder (unidirectional)
    model.add(layers.Dropout(0.8))
    model.add(layers.LSTM(self.hp.h_layers, return_sequences=True, dropout=0.5))
    # Decoder (dense output)
    model.add(layers.Dropout(0.8))
    model.add(layers.TimeDistributed(layers.Dense(self.hp.width_y)))   
    return model 
  
  def __lstm_encoder_decoder(self):
    model = keras.Sequential() 
    # Encoder (bidirectional)
    model.add(layers.Dropout(0.8))
    model.add(layers.LSTM(self.hp.h_layers, input_shape=(self.hp.steps_b, self.hp.width_x), dropout=0.5))
    model.add(layers.RepeatVector(self.hp.steps_f))    
    # Decoder (unidirectional)
    model.add(layers.Dropout(0.8))
    model.add(layers.LSTM(self.hp.h_layers, return_sequences=True, dropout=0.5))
    # Decoder (dense output)
    model.add(layers.Dropout(0.8))
    model.add(layers.TimeDistributed(layers.Dense(self.hp.width_y)))   
    return model 

  
  def __get_arquitetura_compilada(self):

    if(self.hp.arq == ARQ_ENC_DEC):
      model = self.__lstm_encoder_decoder()
    elif(self.hp.arq == ARQ_ENC_DEC_BID):
      model = self.__lstm_encoder_decoder_bidireccional()
    else:
      raise Exception(f"Uma arquitetura desconhecida foi solicitada. Esperava-se [\"{ARQ_ENC_DEC}\", \"{ARQ_ENC_DEC_BID}\"] -> recebido: \"{self.hp.arq}\"")

    model.compile(
      loss=self.hp.error_f, 
      optimizer='nadam', 
      metrics=[self.hp.error_f]
    )
    return model
  #endregion

  #region RELATÓRIOS ESTATÍSTICOS
  def __calcula_stats_e_salva(self, history, XY_train, XY_test, early_stopping_monitor):
    parada              = early_stopping_monitor.stopped_epoch
    _, test_error       = self.modelo.evaluate(XY_test[0], XY_test[1], verbose=0)
    val_error_parada    = history.history[f'val_{self.hp.error_f}'][parada]
    train_error_parada  = history.history[self.hp.error_f][parada]
    
    stat_dict = {
      "Nome"            : self.nome,
      "Época parada"    : parada,
      "Função de erro"  : self.hp.error_f,
      "Erro de treino"  : "{:.4f}".format(train_error_parada),
      "Erro de valid"   : "{:.4f}".format(val_error_parada),
      "Erro de teste"   : "{:.4f}".format(test_error),
    }
    self.stats.append(stat_dict)
    self.print_if_debug(stat_dict)

    # Salva estatísticas em csv
    with open(self.stat_csv_path, 'w') as f:
      w = csv.DictWriter(f, stat_dict.keys())
      w.writeheader()
      w.writerow(stat_dict)
      
    # Salva gráficos
    fg, ax = plt.subplots( nrows=1, ncols=2 ) 
    ax[0].plot(history.history[self.hp.error_f],          label=f'{self.hp.error_f} de treino')
    ax[0].plot(history.history[f'val_{self.hp.error_f}'], label=f'{self.hp.error_f} de validação')
    ax[0].set_xlim([0, EPOCHS])
    ax[0].set_ylim([0.15, 1.7])
    ax[0].legend()
    ax[1].text(0, 0, " "+ str(stat_dict).replace("{","").replace("}","").replace("'","").replace("\"","").replace(",", "\n") +"\n")
    fg.savefig(self.stat_pdf_path, bbox_inches='tight')
    fg.savefig(self.stat_png_path, bbox_inches='tight')

  def evaluate_model(self, Y_true_arg, y_predicted_arg):
    scores_mae    = [] 
    scores_rmse   = []
    scores___smape  = []
    scores_ac     = [] 
    ac_cat        = [] 
    ac_pluv       = []
    r_2           = []
    for i in range(Y_true_arg.shape[1]):
      Y_true      =  [(float(y[i]) if y[i] is not None else float(EPSLON)) for y in Y_true_arg]
      y_predicted =  [(float(y[i]) if y[i] is not None else float(EPSLON)) for y in y_predicted_arg]  

      scores_mae.append( __formata_2_casas(float( mae(Y_true, y_predicted))))
      scores_rmse.append(__formata_2_casas(float(rmse(Y_true, y_predicted))))
      s = __formata_2_casas(float(__smape(Y_true, y_predicted)))
      scores___smape.append(float(s))
      scores_ac.append(__formata_2_casas(float(100-s)))   
      ac_cat.append(None)
      ac_pluv.append(AcuraciaChuvaUtil.get_acuracia_distribuicao_diaria(Y_true, y_predicted, 0, 24, self.hp.steps_b, self.hp.steps_f) if i == 5 else None) 
      r_2.append(__formata_2_casas(100*r2_score(Y_true, y_predicted)) if i != 5 else None)
      if(i==4):
        obj_testagem      = AcuraciaPluvUtil.get_acuracia_distribuicao_diaria(Y_true, y_predicted, 0, 24,  self.hp.steps_b,  self.hp.steps_f, True)  
        scores_mae[-1]    = __formata_2_casas(obj_testagem["mae"])
        scores_rmse[-1]   = __formata_2_casas(obj_testagem["rmse"])
        ac_pluv[-1]       = __formata_2_casas(obj_testagem["ac_cat"]) 
        r_2[-1]           = __formata_2_casas(obj_testagem["r2"]*100)
        s                 = __formata_2_casas(obj_testagem["__smape"])
        scores___smape[-1]  = s
        scores_ac[-1]     = __formata_2_casas(100-s) 
    return [{
      "i"       : i,
      "mae"     : scores_mae[i],
      "rmse"    : scores_rmse[i],
      "__smape"   : scores___smape[i],
      "ac"      : scores_ac[i],
      "ac_cat"  : ac_cat[i],
      "r_2"     : r_2[i],
      "ac_pluv" : ac_pluv[i],
      } for i in range(len(scores_mae))]

  #endregion
  
  #region TREINAMENTO
  def treinar(self, dados_form, iteracoes_teste=I_TESTE_PADRAO):  
    # Fast fail ou cria modelo
    if(self.only_prev):
      raise Exception("Esta é uma instância apenas de previsão, não é permitido: Retreinar; Ressalvar modelo/scalers.")

    self.modelo = self.__get_arquitetura_compilada()

    # Pré processamento
    XY_train  = self.gera_pre_proc_XY(dados_form, iteracoes_teste, True)[1] # XY_train estará em [1]

    # Early stopper (ótima estratégia de regularização)
    early_stopping_monitor = EarlyStopping(
      monitor   = f'val_{self.hp.error_f}', 
      patience  = PATIENCE, 
      verbose   = 1 if self.debug else 0, 
      mode      ='auto', 
      # Nunca usar restore_best_weights = True
      restore_best_weights = False
    )
    
    # Checkpointer a ser chamado pelo early stopper.
    checkpointer = ModelCheckpoint(
      filepath  = self.checkpointed_model_path, 
      monitor   = f'val_{self.hp.error_f}', 
      mode      = 'auto', 
      verbose   = 1 if self.debug else 0, 
      save_best_only    = True,
      # Salva arquitetura junto, não apenas os pesos.
      save_weights_only = False
    )

    # Treina a arquitetura criada
    print(XY_train[0].shape)
    print(XY_train[1].shape)
    history = self.modelo.fit(
      x                = XY_train[0], # X
      y                = XY_train[1], # Y
      validation_split = 0.15,        # 16% de [2014, 2019] na base clima_bsb => 2019
      batch_size       = BATCH_SIZE,
      epochs           = EPOCHS, 
      shuffle          = False,  
      callbacks        = [early_stopping_monitor, checkpointer], 
      verbose          = 2 if self.debug else 0,
    )
  
    # Temos um CHECKPOINTER que persistiu o melhor modelo -> Precisamos apenas recuperá-lo.
    self.modelo = keras.models.load_model(self.checkpointed_model_path)
    
    # Gera relatórios estatísticos do treinamento
    self.__calcula_stats_e_salva(history, XY_train, XY_train, early_stopping_monitor)
  #endregion

  #region PREVISÕES
  def prever(self, dados_form, iteracoes_teste=I_TESTE_PADRAO, inclui_compostas=None, compostas_args=None):     
    df_XY, _, test_XY = self.gera_pre_proc_XY(dados_form, iteracoes_teste) 
    self.print_if_debug(f"Modelo carregado do disco \n SUMÁRIO DE MODELO: {self.modelo.summary()}\n X_test shape = {test_XY[0].shape}")

    # Prepara prev
    base_prev_x = np.array([df_XY[0][-self.hp.steps_b:,:]])
    score = self.modelo.evaluate(
      x = test_XY[0], 
      y = test_XY[1], 
      verbose = 0 if self.debug else 1
    )

    # Debug
    self.print_if_debug(f"{self.modelo.metrics_names[1]}: {score[1]}" )
    self.print_if_debug(f"ÚLTIMAS 24 USADAS S/ INVERSE SCALING: \n {[el for el in base_prev_x[:, :, :self.hp.width_x]]}\n")
    self.print_if_debug(f"ÚLTIMAS 24 USADAS C/ INVERSE SCALING: \n {[self.scalers_x.inverse_transform(el) for el in base_prev_x[:, :, :self.hp.width_x]]}\n")  

    df_pred       = self.modelo.predict(base_prev_x[-1:])
    prev          = np.array([self.scalers_y.inverse_transform(el) for el in df_pred]).reshape(-1, self.hp.width_y)
    
    # Prepara teste
    df_pred_test  = self.modelo.predict(test_XY[0]) 
    prev_test     = np.array([self.scalers_y.inverse_transform(el) for el in df_pred_test]).reshape(-1, self.hp.width_y)
    Y_true_test   = np.array(self.scalers_y.inverse_transform(test_XY[1].reshape(-1, self.hp.width_y)))
  
    # Pós processamento
    if(inclui_compostas is not None):
      prev          = inclui_compostas(prev,        compostas_args)
      prev_test     = inclui_compostas(prev_test,   compostas_args)
      Y_true_test   = inclui_compostas(Y_true_test, compostas_args) 
    
    return self.prev
  #endregion 

#endregion    