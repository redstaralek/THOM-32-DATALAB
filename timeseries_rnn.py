#region IMPORTS
from tensorflow.keras.models         import *
from tensorflow                      import keras
from tensorflow.keras                import layers
from tensorflow.keras.callbacks      import EarlyStopping, ModelCheckpoint 
from sklearn.preprocessing           import RobustScaler
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import mean_squared_error as mse
import gc, joblib, csv, math, pandas as pd, numpy as np, os
from matplotlib import pyplot as plt
#endregion


#region ======================= AUXILIARES ==========================
EPSLON          = 0.0000001
RND_ST          = 142
I_TESTE_PADRAO  = 24
ARQ_ENC_DEC     = "ENCDEC"
ARQ_ENC_DEC_BID = "ENCDEC_BID"
EPOCHS          = 500
PATIENCE        = 50
def cria_diretorio_se_nao_existe(diretorio):
  if not os.path.exists(diretorio):
    os.makedirs(diretorio)
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
    cria_diretorio_se_nao_existe(diretorio)
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
    cria_diretorio_se_nao_existe(diretorio)
    cria_diretorio_se_nao_existe(f'{diretorio}/checkpointed_model')
    cria_diretorio_se_nao_existe(f'{diretorio}/scalers')
    cria_diretorio_se_nao_existe(f'{diretorio}/relatorio')
    self.diretorio        = diretorio
    self.nome             = diretorio.split("__modelos")[-1]
    self.debug            = debug
    self.checkpoint_path  = f"{diretorio}/checkpointed_model"
    self.scalers_x_path   = f'{diretorio}/scalers/scalers_x.gz'
    self.scalers_y_path   = f'{diretorio}/scalers/scalers_y.gz'
    self.hp_dict_path     = f'{diretorio}/params.npy'
    self.stat_path        = f'{diretorio}/relatorio/relatorio'
    self.dataset_path     = f'{diretorio}/relatorio/relatorio_dataset'
    self.t_dataset_path   = f'{diretorio}/relatorio/relatorio_dataset_transformado'
    self.batch_size       = batch_size
    self.early_stopper    = None # Empty at both cases, but datalab instance will eventually populate and use it.

    if(hp is not None):
      # Se forneceu hp, é uma instância de treinamento (uso lab, apenas)
      self.only_prev = False
      # Hiperparâmetros
      self.hp        = hp
      self.hp.salvar(self.diretorio)
      # Modelo, scalers e estatísticas serão todos gerados. Inicialmente vazios/nulos
      self.scalers_x = None
      self.scalers_y = None
      self.modelo    = None
      self.stat_dict = None
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
      # Recupera scalers, model e estatísticas do diretório 
      self.scalers_x = joblib.load(self.scalers_x_path)
      self.scalers_y = joblib.load(self.scalers_y_path)
      self.modelo    = self.__get_arquitetura_compilada()
      self.modelo    = keras.models.load_model(self.checkpoint_path)
      self.stat_dict = np.load(self.stat_path+".npy", allow_pickle=True).item()

  #region AUXILIARES
  def print_if_debug(self, args):
    if(self.debug):
      print(args)
  #endregion

  #region PRÉ-PROCESSAMENTO
  
  def salva_distribuicao(self, dataset, path):
    # Uma linha da img p/ cada grandeza ("column"). Uma coluna da img p/ descrição
    fg, ax = plt.subplots( nrows = len(dataset.columns), ncols=2, figsize=(11, 20), gridspec_kw={'width_ratios':[4, 1]}) 

    for i, col in enumerate(dataset.columns.values):
      p = dataset[col].copy()
      ax[i, 0].hist(p, label = col, bins=600)
      ax[i, 0].legend()
      ax[i, 1].text(0,0,"\n"+str(pd.DataFrame(p).describe())+"\n")
      ax[i, 1].set_xticks([])
      ax[i, 1].set_yticks([])
    fg.savefig(path)

  
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
    tX, tY = [], []

    if(treinamento):  
      self.scalers_x = RobustScaler()
      self.scalers_y = RobustScaler()
      tX = self.scalers_x.fit_transform(X)
      tY = self.scalers_y.fit_transform(Y)

      self.print_if_debug(f"SUMÁRIO DADOS NORMAIS:\n {pd.DataFrame(X).describe()}")
      self.print_if_debug(f"SUMÁRIO DADOS TRANSFORM:\n {pd.DataFrame(tX).describe()}")
      self.salva_distribuicao(X,  self.dataset_path+".pdf")
      self.salva_distribuicao(pd.DataFrame(tX), self.t_dataset_path+".pdf")

      joblib.dump(self.scalers_x, self.scalers_x_path)
      joblib.dump(self.scalers_y, self.scalers_y_path) 
      self.print_if_debug("\n SCALERS SALVOS NO DISCO!\n")  
    else:
      tX = self.scalers_x.transform(X)
      tY = self.scalers_y.transform(Y)

    janela_X, janela_Y = self.to_supervised(tX, tY) 
    test_ratio = n_tests/len(janela_X) if n_tests is not None and n_tests != 0 else None
    janela_X_train, janela_X_test = train_test_split(janela_X, test_size = test_ratio, shuffle = False, random_state = RND_ST) 
    janela_Y_train, janela_Y_test = train_test_split(janela_Y, test_size = test_ratio, shuffle = False, random_state = RND_ST) 

    return [tX, tY], [janela_X_train, janela_Y_train], [janela_X_test, janela_Y_test]

  def __substitui_nulos_e_nan(self, df):
    df = df.copy()
    for grandeza in df.columns.values:
      if(df[grandeza].dtypes != 'float'):
        self.print_if_debug(f'{grandeza} não é float (ignorado).')
        continue
      else:
        media = float(df[grandeza].mean())
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
  def __calcula_stats_e_salva(self, history, XY_t_test):
    parada              = self.early_stopper.stopped_epoch
    # Modelo na parada
    val_error_parada    = history.history[f'val_{self.hp.error_f}'][parada-1]
    train_error_parada  = history.history[self.hp.error_f][parada-1]

    # Melhor modelo (no mínimo de validation error)
    hist_val            = history.history[f'val_{self.hp.error_f}']
    melhor_epoca        = hist_val.index(min(hist_val)) + 1
    val_error_melhor    = history.history[f'val_{self.hp.error_f}'][melhor_epoca-1]
    train_error_melhor  = history.history[self.hp.error_f][melhor_epoca-1]
    _, test_error       = self.modelo.evaluate(XY_t_test[0], XY_t_test[1], verbose=0)
    
    # Prepara dados de teste NÃO TRANSFORMADO
    rmses_str = ""
    rmses     = []
    X_test    = np.array([self.scalers_x.inverse_transform(x)  for x  in XY_t_test[0]])
    Y_test    = np.array([self.scalers_y.inverse_transform(y)  for y  in XY_t_test[1]])
    Y_pred    = np.array([self.scalers_y.inverse_transform(yp) for yp in self.modelo.predict(XY_t_test[0])])
    for i, grandeza in enumerate(self.hp.grandezas[1]):
      Y_test_plano = Y_test[:,:,i].reshape(-1)
      Y_pred_plano = Y_pred[:,:,i].reshape(-1)
      _rmse = math.sqrt(mse(Y_test_plano, Y_pred_plano))
      rmses_str += f"\n  - RMSE p/ \"{grandeza}\": {'{:.4f}'.format(_rmse)}"
      rmses.append('{:.4f}'.format(_rmse))

    # Formata estatísticas: .txt e dictionary p/ [.npy, .csv]
    stat_str = (f"{self.nome}\n"
    + f"\nÉpoca de parada: {str(parada)}"
    + f"\nMelhor época (validação): {str(melhor_epoca)}"
    + f"\nFunção de erro p/ treino: {self.hp.error_f}"
    + f"\nFunção de erro - valores gerais (SCALED) \n"
    + f"\n{self.hp.error_f.title()} trein. (última  época): {'{:.4f}'.format(train_error_parada)}"
    + f"\n{self.hp.error_f.title()} valid. (última  época): {'{:.4f}'.format(val_error_parada)}"
    + f"\n{self.hp.error_f.title()} trein. (melhor modelo): {'{:.4f}'.format(train_error_melhor)}"
    + f"\n{self.hp.error_f.title()} valid. (melhor modelo): {'{:.4f}'.format(val_error_melhor)}"
    + f"\n{self.hp.error_f.title()} teste  (melhor modelo): {'{:.4f}'.format(test_error)}"
    + f"\n"
    + f"\nTeste RMSE - val. por grandeza (UNSCALED) \n{rmses_str}")

    self.stat_dict = {
      "nome"                        : self.nome,
      "parada"                      : parada,
      "melhorEpoch"                 : melhor_epoca,
      "erro_treino_parada"          : '{:.4f}'.format(train_error_parada),
      "erro_valid_parada"           : '{:.4f}'.format(val_error_parada),
      "erro_treino_melhor"          : '{:.4f}'.format(train_error_melhor),
      "erro_valid_melhor"           : '{:.4f}'.format(val_error_melhor),
      "erro_teste_melhor"           : '{:.4f}'.format(test_error),
      "rmse_unscaled_por_grandeza"  : rmses
    }

    # Salva estatísticas em .csv e .npy
    with open(f'{self.stat_path}.csv', 'w') as f:
      w = csv.DictWriter(f, self.stat_dict.keys())
      w.writeheader()
      w.writerow(self.stat_dict)
    np.save(f"{self.stat_path}.npy", self.stat_dict)
    gc.collect()
      
    # Salva gráficos
    fg, ax = plt.subplots( nrows=1, ncols=2, figsize=(8, 5), gridspec_kw={'width_ratios':[1, 2]}) 
    ax[0].plot(history.history[self.hp.error_f],          label=f'{self.hp.error_f} de treino')
    ax[0].plot(history.history[f'val_{self.hp.error_f}'], label=f'{self.hp.error_f} de validação')
    ax[0].scatter([melhor_epoca], [val_error_melhor], marker='*', c='r', zorder=3, s=35)
    ax[0].legend()
    ax[1].text(0, 0, str(stat_str) +"\n")
    
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_axis_off()
    fg.savefig(self.stat_path+".pdf", bbox_inches='tight')
    fg.savefig(self.stat_path+".png", bbox_inches='tight')
  #endregion
  
  #region TREINAMENTO
  def treinar(self, dados_form, n_tests=I_TESTE_PADRAO):  
    # Fast fail ou cria modelo
    if(self.only_prev):
      raise Exception("Esta é uma instância apenas de previsão, não é permitido: Retreinar; Ressalvar modelo/scalers.")

    self.modelo = self.__get_arquitetura_compilada()

    # Pré processamento
    XY_all_plain, XY_j_train, XY_j_test  = self.gera_pre_proc_XY(dados_form, n_tests, True)

    # Early stopper (ótima estratégia de regularização)
    self.early_stopper = EarlyStopping(
      monitor   = f'val_{self.hp.error_f}', 
      patience  = PATIENCE, 
      verbose   = 1 if self.debug else 0, 
      mode      = 'auto', 
      # Nunca usar restore_best_weights = True
      restore_best_weights = False
    )
    
    # Checkpointer a ser chamado pelo early stopper.
    checkpointer = ModelCheckpoint(
      filepath  = self.checkpoint_path, 
      monitor   = f'val_{self.hp.error_f}', 
      mode      = 'auto', 
      verbose   = 1 if self.debug else 0, 
      save_best_only    = True,
      # Salva arquitetura junto, não apenas os pesos.
      save_weights_only = False
    )

    # Treina a arquitetura criada
    history = self.modelo.fit(
      x                = XY_j_train[0], # X
      y                = XY_j_train[1], # Y
      validation_split = 0.15,        # 16% de [2014, 2019] na base clima_bsb => 2019
      batch_size       = self.batch_size,
      epochs           = EPOCHS, 
      shuffle          = False,  
      callbacks        = [self.early_stopper, checkpointer], 
      verbose          = 2 if self.debug else 0,
    )
  
    # Temos um CHECKPOINTER que persistiu o melhor modelo -> Precisamos apenas recuperá-lo.
    self.modelo = keras.models.load_model(self.checkpoint_path)
    
    # Gera relatórios estatísticos do treinamento
    self.__calcula_stats_e_salva(history, XY_j_test)
  #endregion

  #region PREVISÕES
  def prever(self, XY_dict):  
    '''
    Gera previsão com base em um vetor de entrada.
    
    - XY_dict:  Array de dicionários das grandezas
    - debug:    Ativa verbose para todos os processos dessa instância.

    -     return: Retorna vetor de 3 posições.
      - 1º: Vetor no formato [Y]. Janela única de previsão.         Shape = (1, STEPS_FORWARD)
    '''   
    tXY, _, _ = self.gera_pre_proc_XY(XY_dict, 0) 
    X, _ = tXY
    self.print_if_debug(f"Modelo carregado do disco \n SUMÁRIO DE MODELO: {self.modelo.summary()}\n")

    # Prevê próximas leituras com base na última janela
    base_prev_x   = np.array([X[-self.hp.steps_b:,:]])
    df_pred       = self.modelo.predict(base_prev_x[-1:])
    prev          = np.array([self.scalers_y.inverse_transform(el) for el in df_pred]).reshape(-1, self.hp.width_y)

    # Útil para verificar se está usando o intervalo correto p/ prever
    self.print_if_debug(f"ÚLTIMAS H USADAS: \n {[self.scalers_x.inverse_transform(el) for el in base_prev_x[-1:]]}\n")
  
    return prev
  #endregion 

#endregion    