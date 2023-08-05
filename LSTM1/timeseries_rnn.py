from statsmodels.tools.eval_measures import rmse
from sklearn.metrics                 import r2_score, mean_absolute_error as mae 
from tensorflow.keras.models         import model_from_json,  model_from_json
from tensorflow                      import keras
from tensorflow.keras                import layers
from tensorflow.keras.callbacks      import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing           import RobustScaler
from sklearn.model_selection         import train_test_split
import os, gc, joblib, pandas as pd, numpy as np
from utils import *
from inicializacao import *
from matplotlib import pyplot as plt

MODELO_PASTA = '__arquivos_rnn'
MODELO_SUBPASTAS = os.listdir(MODELO_PASTA)
STEPS_B, STEPS_F = 24, 24

class rnn_aux:

  @staticmethod
  def compila(model): 
    model.compile(loss='mse', optimizer='nadam', metrics=['mse']) 
    return model
  
  @staticmethod
  def carrega_e_compila(nome): 
      loaded_model = rnn_aux.carrega_modelo(nome)
      loaded_model.compile(   
          loss='mse',
          optimizer='nadam',   
          metrics=['mse'], 
          ) 
      return loaded_model

  @staticmethod
  def salva_modelo(model, nome): 
    model_json = model.to_json()
    with open(f"{nome}.json", "w") as json_file:
        json_file.write(model_json) 
    model.save_weights(f"{nome}.h5") 
    print("Saved model to disk")

  @staticmethod
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
   
  @staticmethod
  def pre_proc(dados_form, treinamento_e_salva_scalers, modelo_index=0):
    #################################################################### PRÉ PROCESSAMENTO ################################################################## 
    df_input = pd.DataFrame(dados_form)
    y_columns =   ["temp",             "hum",             "pres",        "rad",        
                   "pluv"]    
    x_columns =   ["temp",             "hum",             "pres",        "rad",   
                   "pluv",             "dia_ano",         "horario"]     

    df_input_x, df_input_y, scalers_y, scalers_x = [], [], RobustScaler(), RobustScaler()
    el_size_y, el_size_x = len(y_columns), len(x_columns)

    diretorio_nome = MODELO_SUBPASTAS[modelo_index]
    
    if(treinamento_e_salva_scalers):  
      df_input_x = scalers_x.fit_transform(df_input[x_columns]) 
      df_input_y = scalers_y.fit_transform(df_input[y_columns]) 

      print("\n SALVANDO SCALERS NO DISCO!\n") 
      joblib.dump(scalers_x, f'__arquivos_rnn/{diretorio_nome}/scalers_x.gz')
      joblib.dump(scalers_y, f'__arquivos_rnn/{diretorio_nome}/scalers_y.gz') 
      print("\n SCALERS SALVOS NO DISCO!\n")  
    else:
      scalers_x = joblib.load(f'__arquivos_rnn/{diretorio_nome}/scalers_x.gz')
      scalers_y = joblib.load(f'__arquivos_rnn/{diretorio_nome}/scalers_y.gz')
      print(f"1 x scaler={scalers_x}, x={df_input_x}")
      print(f"1 y scaler={scalers_y}, y={df_input_y}")
      df_input_x = scalers_x.transform(df_input[x_columns]) 
      df_input_y = scalers_y.transform(df_input[y_columns]) 
      print(f"2 x scaler={scalers_x}, x={df_input_x}")
      print(f"2 y scaler={scalers_y}, y={df_input_y}")
 
    base_x, base_y = rnn_aux.to_supervised(df_input_x, df_input_y) 
    
    return base_x, base_y, df_input_x, df_input_y,  el_size_y, el_size_x, scalers_x, scalers_y 

  
  @staticmethod
  def to_supervised(x_input, y_input):
    x, y         = [], []  
    #loop de dias => supervised com janela móvel de 24h
    for i in range(STEPS_B, len(x_input) - STEPS_F):   
      _x_aux, _y_aux = np.array(x_input[i - STEPS_B : i]), np.array(y_input[i : i + STEPS_F])
      if(len(_x_aux) > 0):
        x.append(_x_aux) 
        y.append(_y_aux)     

    return np.array(x), np.array(y)

  @staticmethod
  def evaluate_model(Y_true_arg, y_predicted_arg):
    scores_mae    = [] 
    scores_rmse   = []
    scores_smape  = []
    scores_ac     = [] 
    ac_cat        = [] 
    ac_pluv       = []
    r_2           = []
    for i in range(Y_true_arg.shape[1]):
      Y_true      =  [(float(y[i]) if y[i] is not None else float(0.001)) for y in Y_true_arg]
      y_predicted =  [(float(y[i]) if y[i] is not None else float(0.001)) for y in y_predicted_arg]  

      scores_mae.append( formata_2_casas(float( mae(Y_true, y_predicted))))
      scores_rmse.append(formata_2_casas(float(rmse(Y_true, y_predicted))))
      s = formata_2_casas(float(smape(Y_true, y_predicted)))
      scores_smape.append(float(s))
      scores_ac.append(formata_2_casas(float(100-s)))   
      ac_cat.append(None)
      ac_pluv.append(AcuraciaChuvaUtil.get_acuracia_distribuicao_diaria(Y_true, y_predicted, 0, 24, STEPS_B, STEPS_F) if i == 5 else None) 
      r_2.append(formata_2_casas(100*r2_score(Y_true, y_predicted))                                                   if i != 5 else None) 

      if(i==4):
        obj_testagem      = AcuraciaPluvUtil.get_acuracia_distribuicao_diaria(Y_true, y_predicted, 0, 24, STEPS_B, STEPS_F, True)  
        scores_mae[-1]    = formata_2_casas(obj_testagem["mae"])
        scores_rmse[-1]   = formata_2_casas(obj_testagem["rmse"])
        ac_pluv[-1]       = formata_2_casas(obj_testagem["ac_cat"]) 
        r_2[-1]           = formata_2_casas(100*obj_testagem["r2"])
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
 
  @staticmethod
  def retorna_prev_e_erros(prev_test, Y_true, prev):
    scores      = rnn_aux.evaluate_model(Y_true, prev_test) 
    prev_finais = []
    # intervalos  = []
    prev_list   = prev[-STEPS_F:].tolist()
    for i in range(prev.shape[1]):
      prev_el = [(float(el[i]) if el[i] is not None else float(0.001)) for el in prev_list]
      prev_finais.append(prev_el) 
      # intervalos.append(rnn_aux.intervalo_confianca_generico(prev_el, scores[i]["rmse"])) 
    return {
      "prev":       prev_finais,
      # "intervalos": intervalos,
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


def treina_rnn(dados_form, modelo_index=0):     

  X, Y, _, _, el_size_y, el_size_x, _, _ = rnn_aux.pre_proc(dados_form, True, modelo_index) 

  # ##################################################################### LSTM MODEL
  early_stopping_monitor = EarlyStopping(monitor='mse', patience=100, verbose=1, mode='auto')
  checkpointer = ModelCheckpoint(filepath = f"__arquivos_rnn/{MODELO_SUBPASTAS[modelo_index]}/checkpoint", verbose=0, save_best_only=True)
  HIDDEN_LAYERS = 500

  model = keras.Sequential() 
  # Encoder (bidirectional)
  model.add(layers.Dropout(0.8))
  model.add(layers.Bidirectional(
    layers.LSTM(HIDDEN_LAYERS, input_shape=(STEPS_B, el_size_x), dropout=0.5)
  ))
  # Enc -> Dec
  model.add(layers.RepeatVector(STEPS_F))    
  # Decoder (unidirectional)
  model.add(layers.LSTM(HIDDEN_LAYERS, return_sequences=True, dropout=0.5))
  model.add(layers.Dropout(0.5))
  model.add(layers.TimeDistributed(
    layers.Dense(el_size_y, activation=layers.LeakyReLU(alpha=0.01))
  ))   

  model = rnn_aux.compila(model)  
  history = model.fit(
    X, 
    Y,    
    validation_split = 0.2,
    batch_size = 1024,
    epochs = 1000, 
    shuffle = False,  
    callbacks=[early_stopping_monitor, checkpointer], 
    verbose=2,
  )
  
  #Salva o modelo
  modelo_index = modelo_index if modelo_index is not None else 0
  rnn_aux.salva_modelo(model, f"__arquivos_rnn/{MODELO_SUBPASTAS[modelo_index]}/model_final")
  print(early_stopping_monitor.stopped_epoch)

  # plot training history
  plt.plot(history.history['mse'], label='train (MSE)')
  plt.plot(history.history['val_mse'], label='test (MSE)')
  plt.legend()
  plt.show()

   
def previsao_rnn(dados_form, iteracoes_teste= 24, modelo_index=0):     

  X, Y, df_input_x, df_input_y, el_size_y, el_size_x, scalers_x, scalers_y = rnn_aux.pre_proc(dados_form, False, modelo_index) 

  iteracoes_teste
  test_ratio = iteracoes_teste/len(X) 
  print(f"\nTEST RATIO ==>{iteracoes_teste} / {len(X)} = {test_ratio}\n")
  _, X_test = train_test_split(X, test_size=test_ratio, shuffle=False, random_state=142) 
  _, Y_test = train_test_split(Y, test_size=test_ratio, shuffle=False, random_state=142) 
  
  diretorio = MODELO_SUBPASTAS[modelo_index]
  modelo = rnn_aux.carrega_e_compila(f'__arquivos_rnn/{diretorio}/model_final')
  print(f"Modelo carregado do disco \n SUMÁRIO DE MODELO: {modelo.summary()}\n X_test shape = {X_test.shape}")
  score = modelo.evaluate(X_test, Y_test, verbose=0)
  print(f"{modelo.metrics_names[1]}: {score[1]}" )

  #--------------- prepara prev ---------------
  base_prev_x = np.array([df_input_x[-STEPS_B:,:]])
  # print(f"ÚLTIMOS 24 X USADOS (SCALED): \n {np.array([el for el in base_prev_x[:,:,:el_size_x]])}\n")
  # print(f"ÚLTIMOS 24 X USADOS (BRUTOS): \n {np.array([scalers_X.inverse_transform(el) for el in base_prev_x[:,:,:el_size_x]])}\n")  
  base_prev_y = np.array([df_input_y[-STEPS_B:,:]])
  print(f"ÚLTIMOS 24 Y USADOS (SCALED): \n {np.array([el for el in base_prev_y[:,:,:el_size_y]])}\n")
  print(f"ÚLTIMOS 24 Y USADOS (BRUTOS): \n {np.array([scalers_y.inverse_transform(el) for el in base_prev_y[:,:,:el_size_y]])}\n")  

  df_pred       = modelo.predict(base_prev_x[-1:])
  prev          = np.array([scalers_y.inverse_transform(el) for el in df_pred]).reshape(-1,el_size_y)
  
  #--------------- prepara teste --------------- 
  df_pred_test  = modelo.predict(X_test) 
  prev_test     = np.array([scalers_y.inverse_transform(el) for el in df_pred_test]).reshape(-1,el_size_y)
  Y_true_test   = np.array(scalers_y.inverse_transform(Y_test.reshape(-1,el_size_y))) 
 
  return rnn_aux.retorna_prev_e_erros(prev_test, Y_true_test, prev, STEPS_B, STEPS_F)
   