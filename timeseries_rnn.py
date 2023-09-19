#region IMPORTS
from tensorflow.keras.models         import *
from tensorflow                      import keras
from tensorflow.keras                import layers
from tensorflow.keras.callbacks      import EarlyStopping, ModelCheckpoint 
from sklearn.preprocessing           import StandardScaler
from sklearn.model_selection         import train_test_split
import os, gc, joblib, csv, pandas as pd, numpy as np
from matplotlib import pyplot as plt
#endregion


#region ======================= AUXILIARES ==========================
EPSLON          = 0.0000001
RND_ST          = 142
I_TESTE_PADRAO  = 24
ARQ_ENC_DEC     = "enc_dec"
ARQ_ENC_DEC_BID = "enc_dec_b"
EPOCHS          = 200
PATIENCE        = 25
#endregion


#region ================ CLASSE DE HIPERPARÂMETROS ==================
class MZDN_HP:
  def __init__(self, grandezas, error_f, h_layers, steps, arq=None):
    self.grandezas              = grandezas # [0] contém X e [1] contém Y
    self.width_x, self.width_y  = len(self.grandezas[0]), len(self.grandezas[1])
    self.steps_b, self.steps_f  = steps , 24
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
  
  def __init__(self, diretorio, hp=None, debug=True, batch_size=1024):
    '''
    Construtor

    - diretorio: Diretório a buscar ou salvar dados do modelo e seus hiperparâmetros.
    - hp: Instância de hiperparâmetros. Para construir instâncias de TESTE. Outrocaso, hp é recuperado do diretório.
    - debug: Ativa verbose para todos os processos dessa instância.

    -     return: None
    '''
    self.diretorio = diretorio
    self.nome      = diretorio.split("__modelos")[-1]
    self.debug     = debug
    self.stats     = []
    self.checkpointed_model_path = f"{diretorio}/checkpointed_model"
    self.scalers_x_path          = f'{diretorio}/scalers/scalers_x.gz'
    self.scalers_y_path          = f'{diretorio}/scalers/scalers_y.gz'
    self.hp_dict_path            = f'{diretorio}/params.npy'
    self.stat_csv_path           = f'{diretorio}/relatorio/relatorio.csv'
    self.stat_pdf_path           = f'{diretorio}/relatorio/relatorio.pdf'
    self.stat_png_path           = f'{diretorio}/relatorio/relatorio.png'
    self.batch_size              = batch_size

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
                        hp_dict["error_f"],
                        hp_dict["h_layers"],
                        hp_dict["steps_b"],
                        hp_dict["arq"])
      # Recupera scalers e model do diretorio 
      self.scalers_x = joblib.load(self.scalers_x_path)
      self.scalers_y = joblib.load(self.scalers_y_path)
      self.modelo    = self.__get_arquitetura_compilada()
      self.modelo    = keras.models.load_model(self.checkpointed_model_path)

  #region AUXILIARES
  def print_if_debug(self, args):
    if(self.debug):
      print(args)
  #endregion

  #region PRÉ-PROCESSAMENTO
  def gera_pre_proc_XY(self, XY_dict, n_tests=0, treinamento=False):
    '''
    Pré processa os dados e fornece outputs úteis diversos

    - XY_dict: Dicionário de valores a ser trabalhado
    - n_tests: A quantidade de elementos p/ teste, se desejado. Padrão = 0
    - treinamento: Se vai treinar o modelo e persistir scalers. Senão (caso de previsão), procura e utiliza os scalers do diretório.

    -     return: Retorna vetor de 3 posições.
      - 1º: Vetor no formato [[X], [Y]]. valores PLANOS de X e Y transformados SEM SPLIT.
      - 2º: Vetor no formato [[X], [Y]]. Valores JANELADOS e transformados de TREINO para X e Y.
      - 3º: Vetor no formato [[X], [Y]]. Valores JANELADOS e transformados de TESTE para X e Y. Vazios p/ n_tests não especificado.
    '''
    df = pd.DataFrame(XY_dict).set_index("data")
    X = self.__substitui_nulos_e_nan(df[self.hp.grandezas[0]])
    Y = self.__substitui_nulos_e_nan(df[self.hp.grandezas[1]])
    
    if(treinamento):  
      self.scalers_y, self.scalers_x = StandardScaler(), StandardScaler()
      tX = self.scalers_x.fit_transform(X) 
      tY = self.scalers_y.fit_transform(Y)

      joblib.dump(self.scalers_x, self.scalers_x_path)
      joblib.dump(self.scalers_y, self.scalers_y_path) 
      self.print_if_debug("\n SCALERS SALVOS NO DISCO!\n")  
    else:
      tX = self.scalers_x.transform(X) 
      tY = self.scalers_y.transform(Y)   
  
    self.print_if_debug(f"SUMÁRIO DADOS NORMAIS: {pd.DataFrame(tX).describe()}")
    self.print_if_debug(f"SUMÁRIO DADOS TRANSFORM: {pd.DataFrame(X).describe()}")

    janela_X, janela_Y = self.to_supervised(tX, tY) 
    test_ratio = n_tests/len(janela_X)

    janela_X_train, janela_X_test = train_test_split(janela_X, test_size = test_ratio, shuffle = False, random_state = RND_ST) 
    janela_Y_train, janela_Y_test = train_test_split(janela_Y, test_size = test_ratio, shuffle = False, random_state = RND_ST) 

    return [tX, tY], [janela_X_train, janela_Y_train], [janela_X_test, janela_Y_test]

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
    ax[0].set_ylim([0.15, 1.5])
    ax[0].legend()
    ax[1].text(0, 0, str(stat_dict).replace("{","").replace("}","").replace("'","").replace("\"","").replace(",", "\n").replace("Nome: /", " ") +"\n")
    fg.savefig(self.stat_pdf_path, bbox_inches='tight')
    fg.savefig(self.stat_png_path, bbox_inches='tight')
  #endregion
  
  #region TREINAMENTO
  def treinar(self, dados_form, n_tests=I_TESTE_PADRAO):  
    # Fast fail ou cria modelo
    if(self.only_prev):
      raise Exception("Esta é uma instância apenas de previsão, não é permitido: Retreinar; Ressalvar modelo/scalers.")

    self.modelo = self.__get_arquitetura_compilada()

    # Pré processamento
    XY_train  = self.gera_pre_proc_XY(dados_form, n_tests, True)[1] # XY_train estará em [1]

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
      batch_size       = self.batch_size,
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
  def prever(self, XY_dict, n_tests=I_TESTE_PADRAO):  
    '''
    Gera previsão com base em um vetor de entrada.
    
    - XY_dict:  Array de dicionários das grandezas
    - debug:    Ativa verbose para todos os processos dessa instância.

    -     return: Retorna vetor de 3 posições.
      - 1º: Vetor no formato [Y]. Janela única de previsão.         Shape = (1, STEPS_FORWARD)
    '''   
    tXY, _, jan_XY_test = self.gera_pre_proc_XY(XY_dict, n_tests) 
    self.print_if_debug(f"Modelo carregado do disco \n SUMÁRIO DE MODELO: {self.modelo.summary()}\n X_test shape = {jan_XY_test[0].shape}")

    # Score rápido dos últimos dias definidos pelo split
    score = self.modelo.evaluate(
      x = jan_XY_test[0], 
      y = jan_XY_test[1], 
      verbose = 0 if self.debug else 1
    )
    self.print_if_debug(f"{self.modelo.metrics_names[1]}: {score[1]}" )

    # Prevê próximas leituras com base na última janela
    base_prev_x   = np.array([tXY[0][-self.hp.steps_b:,:]])
    df_pred       = self.modelo.predict(base_prev_x[-1:])
    prev          = np.array([self.scalers_y.inverse_transform(el) for el in df_pred]).reshape(-1, self.hp.width_y)

    # Útil para verificar se está usando o intervalo correto p/ prever
    self.print_if_debug(f"ÚLTIMAS H USADAS: \n {[self.scalers_x.inverse_transform(el) for el in base_prev_x[-1:]]}\n")  
  
    return prev
  #endregion 

#endregion    