from datetime                         import datetime, timedelta
from statsmodels.tools.eval_measures  import rmse
from sklearn.metrics                  import  precision_score, r2_score , mean_absolute_error as mae  
import numpy as np

def formata_2_casas(num): 
  return float('{0:.2f}'.format(num)) if num is not None else None

def smape(A, F):
  A       = np.array(A)
  F       = np.array(F)
  epsilon = 0.1
  resp    = 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + epsilon))
  return resp if resp == resp else 0 #se não NaN

def corta_proximo_cem(array):
    array_final = []
    maximo, minimo = 98, 2
    for elemento in array:
        if(elemento >= maximo ):
            array_final.append(maximo)
        elif(elemento <= minimo ):
            array_final.append(minimo)
        else:
            array_final.append(elemento)
    return array_final


def proc_hum(el):
    el = float(el)
    if(el<99 and el > 5): 
        return el
    elif(el >= 99): 
        return 99
    return 5


def proc_rad_elemento(el):
    return float(el) if el is not None and el >= 0 else 0


class AcuraciaChuvaUtil:
  @staticmethod
  def get_acuracia_distribuicao_diaria(true_data, pred_data, inicio, fim, steps_b, steps_f):
    true_data_pp, pred_data_pp = AcuraciaChuvaUtil.to_supervised(true_data, pred_data, steps_b, steps_f, inicio, fim)  
    x_pos_proc, y_pos_proc = [], []
    for i in range(len(true_data_pp)):
      aux_true, aux_pred = AcuraciaChuvaUtil.classificador_binario_chuva(true_data_pp[i], pred_data_pp[i])
      x_pos_proc.append(aux_true)
      y_pos_proc.append(aux_pred) 
    return float('{0:.2f}'.format(100*precision_score(x_pos_proc, y_pos_proc, labels=[0,1],average="macro")))

  @staticmethod
  def classificador_binario_chuva(a,b):
    prob_nao_chover_a = prob_nao_chover_b = 1
    for i in range(len(a)):  
      prob_nao_chover_a = prob_nao_chover_a*( float(100-a[i])/100 )  
    for i in range(len(b)):   
      prob_nao_chover_b = prob_nao_chover_b*( float(100-b[i])/100 )   
    return (1 if prob_nao_chover_a>0.5 else 0), (1 if prob_nao_chover_b>0.5 else 0)

  @staticmethod
  def to_supervised(true_data, pred_data, steps_b, steps_f,inicio, fim):  
    true_final, pred_final = [], []
    for i in range(0, len(true_data) - steps_f):   
      _true = np.array(true_data[i : i + steps_f]) 
      _pred = np.array(pred_data[i : i + steps_f]) 
      true_final.append(_true[inicio:fim]) 
      pred_final.append(_pred[inicio:fim])
    return np.array(true_final), np.array(pred_final)


class AcuraciaPluvUtil:
  @staticmethod
  def get_acuracia_distribuicao_diaria(true_data, pred_data, inicio, fim, steps_b, steps_f, _print):
    true_data_pp, pred_data_pp = AcuraciaPluvUtil.to_supervised(true_data, pred_data, steps_b, steps_f, inicio, fim)  
    x_pos_proc, y_pos_proc, x_cat_pos_proc, y_cat_pos_proc = [], [], [], []
    for i in range(len(true_data_pp)): 
      x_pos_proc.append(np.sum(true_data_pp[i]))
      y_pos_proc.append(np.sum(pred_data_pp[i]))  
      x_cat_pos_proc.append(0 if x_pos_proc[-1] == 0 else 1)
      y_cat_pos_proc.append(0 if y_pos_proc[-1] == 0 else 1) 
    return {"mae":    float(mae(x_pos_proc, y_pos_proc)),
            "rmse":   float(rmse(x_pos_proc, y_pos_proc)),
            "r2":     float(r2_score(x_pos_proc, y_pos_proc)),
            "smape":  float(smape(x_pos_proc, y_pos_proc)),
            "ac_cat": float('{0:.2f}'.format((100*precision_score(x_cat_pos_proc, y_cat_pos_proc, labels=[0,1],average="macro"))))
            }

  @staticmethod
  def to_supervised(true_data, pred_data, steps_b, steps_f,inicio, fim):  
    true_final, pred_final = [], []
    for i in range(0, len(true_data) - steps_f):
      true_final.append(np.array(true_data[i : i + steps_f])[inicio:fim]) 
      pred_final.append(np.array(pred_data[i : i + steps_f])[inicio:fim])
    return np.array(true_final), np.array(pred_final)
