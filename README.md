# THOM-32-DATALAB
Datalab para treino e avaliação dos modelos de IA



-Utilizar requirements.txt para baixar extensões necessárias.

-Ambiente de desenvolvimento com tensorflow (em GPU) usando Conda
    . conda activate tf2.13.0; 

-Comandos de compilação parcelados:

    python main.py --grandezas 'AX_AY'      --batch_sizes 32    --efunc mae;
    python main.py --grandezas 'ApDX_ApDY'  --batch_sizes 32    --efunc mae;
