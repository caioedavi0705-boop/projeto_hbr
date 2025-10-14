import numpy as np
import pandas as pd

#Importação dos dados de treino
fd001_train = pd.read_csv("train_FD001.txt",
                    sep='\s+',
                    header= None)  

fd001_test = pd.read_csv("test_FD001.txt",
                    sep='\s+',
                    header= None) 

rul001 = pd.read_csv("RUL_FD001.txt",sep='\s+',header= None) 

##Remoção de colunas inteiramente preenchidas por NaN
fd001_train = fd001_train.dropna(axis=1,how='all')
fd001_test = fd001_test.dropna(axis=1,how='all')

##Introdução do cabeçalho do data frame
cabecalho1 = ["ID","Nº Ciclos","Altitude [ft]", "Mach","TRA", "T2 [°R]", "T24 [°R]", "T30 [°R]", 
             "T50[°R]", "P2 [psia]", "P15 [psia]", "P30 [psia]", "Nf [rpm]", "Nc [rpm]", "epr [-]",
             "Ps30 [psia]", "phi [pps/psi]", "NRf [rpm]", "NRc [rpm]", "BPR [-]", "farB [-]",
             "htBleed [-]", "Nf_dmd [rpm]", "PCNfR_dmd [rpm]", "W31 [lbm/s]", "W32 [lbm/s]"] 
fd001_train.columns = cabecalho1 
fd001_test.columns = cabecalho1  
cabecalho2 = ["rul"]
rul001.columns = cabecalho2

print(fd001_train.head())

##Eliminando valores vazios
fd001_train = fd001_train.dropna() 
fd001_test = fd001_test.dropna() 

##Geração de Histogramas e seleção de variáveis 
import seaborn as sns 
import matplotlib.pyplot as plt

fig,axes=plt.subplots(nrows=6,ncols=4,figsize=(20,16))
axes=axes.ravel()
for i,item in enumerate(fd001_train.columns[2:]):
    sns.histplot(fd001_train[item],bins=50,ax=axes[i])
    axes[i].set_title(f'{item}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('') 

plt.tight_layout()
plt.show()

features_train =fd001_train.drop(columns=['TRA','T2 [°R]','P2 [psia]','epr [-]',
                                    'farB [-]', 'PCNfR_dmd [rpm]'])

features_test =fd001_test.drop(columns=['TRA','T2 [°R]','P2 [psia]','epr [-]',
                                    'farB [-]','PCNfR_dmd [rpm]']) 

##Normalização 
from sklearn.preprocessing import MinMaxScaler

features=features_train.columns[2:]
scaler = MinMaxScaler()
features_train[features] = scaler.fit_transform(features_train[features])
features_test[features]=scaler.transform(features_test[features]) 

##Definição dos targets para regressão e classificação nos dados de treino
limit_r = 130 
limit_c = 50
max_ciclos = fd001_train.groupby('ID')['Nº Ciclos'].max().reset_index()
rul_train = []
for m in fd001_train['ID'].unique():
    motor = fd001_train[fd001_train['ID']==m]
    aux=[]
    for k in range(len(motor['Nº Ciclos'])):
        if k+1 <= max_ciclos['Nº Ciclos'].iloc[m-1]-limit_r:
            aux.append(limit_r)
        else:
            aux.append(max_ciclos['Nº Ciclos'].iloc[m-1]-
                       motor['Nº Ciclos'].iloc[k]) 
    rul_train.extend(aux) 
rul_train = [int(x) for x in rul_train]
label_train = [1 if x >= limit_c else 0 for x in rul_train] 

##Visualização de um ciclo de vida de um motor 
exemplo = fd001_train[fd001_train['ID']==1]
l=len(exemplo['Nº Ciclos'])

plt.figure(figsize=(10,6))

plt.plot(exemplo['Nº Ciclos'].iloc[0:l],rul_train[0:l],c='red')
plt.title('Representação do ciclo de vida do motor ID = 1')
plt.ylim(0,140)
plt.xlim(0,200)
plt.xlabel('Ciclo do motor')
plt.ylabel('RUL')

plt.show()

##Definição dos targets para regressão e classificação nos dados de teste 
max_ciclos_test = fd001_test.groupby('ID')['Nº Ciclos'].max().reset_index()
rul_test = []
for m in fd001_test['ID'].unique():
    engine = fd001_test[fd001_test['ID']==m]
    for k in range(len(engine['Nº Ciclos'])):
        rul_aux=(max_ciclos_test['Nº Ciclos'].iloc[m-1]-
                 engine['Nº Ciclos'].iloc[k])
        v = rul_aux+rul001['rul'].iloc[m-1]
        if v >= limit_r:
            rul_test.append(limit_r)
        else:
            rul_test.append(v)
rul_test = [int(x) for x in rul_test]
label_test = [1 if x >= limit_c else 0 for x in rul_test] 

##Criação das janelas de tempo para dados de treino e de teste
def time_window(data,rul,label,window_size,step):
    x,y_r,y_c = [], [], []
    l=0
    for m in data['ID'].unique():
        engine = data[data['ID']==m]
        for i in range(0,len(engine)-window_size+1,step):
            f= i + window_size
            x.append(engine.iloc[i:f,2:].values)
            y_r.append(rul[l+f-1]) 
            y_c.append(label[l+f-1])
        l += len(engine)
    return np.array(x), np.array(y_r), np.array(y_c)
window_size = 30
step = 1
x_train, y_train_r,y_train_c= time_window(features_train,rul_train,
                                                        label_train,window_size,step)

print(x_train.shape,y_train_r.shape,y_train_c.shape)  
x_test, y_test_r,y_test_c = time_window(features_test,rul_test,
                                                        label_test,window_size,step)
print(x_test.shape,y_test_r.shape,y_test_c.shape)

##Separação das variáveis de treino e teste 
from sklearn.model_selection import train_test_split

x1,x2,y1,y2 = train_test_split(x_train,y_train_r,test_size=0.2,random_state=1)
x3,x4,y3,y4 = train_test_split(x_train,y_train_c,test_size=0.2,random_state=1) 

x1=x1.astype('float32')
x2=x2.astype('float32')
x3=x3.astype('float32')
x4=x4.astype('float32')
y1=y1.astype('float32')
y2=y2.astype('float32')
y3=y3.astype('float32')
y4=y4.astype('float32')

##Criação do modelo regressão
from tensorflow import keras 
from keras import layers
from keras.models import Sequential
from keras.metrics import AUC
from keras.layers import LSTM,Dense,Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf 
import keras_tuner
import time
 
shape1 = (x1.shape[1], x1.shape[2])

def build_model_regression(hp):
    model=Sequential()
    n_layers_lstm_r=hp.Int('layers_lstm_r',1,4)
    n_layers_dense_r=hp.Int('layers_dense_r',1,3)
    model.add(LSTM(hp.Int('lstm_r_0',min_value=32,max_value=256,step=32),
                   activation='tanh',
                   return_sequences=True,input_shape=shape1))
    for i in range(n_layers_lstm_r):
        return_seq = i <n_layers_lstm_r -1
        model.add(LSTM(hp.Int(f'lstm_r_{i}',min_value=32,max_value=256,step=32),
                       activation='tanh',
                       return_sequences=return_seq))
    if hp.Boolean("dropout"):
        model.add(Dropout(0.25))
    for i in range(n_layers_dense_r):
        model.add(Dense(hp.Int(f'dense_r_{i+1}',min_value=32,max_value=256,step=32),
                activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer=RMSprop(learning_rate=hp.Choice('learning_rate_r',[0.01,0.001,0.0001])), 
                  loss='mse', 
              metrics=['mean_squared_error','mean_absolute_error'])
    return model 

start=time.time()
tuner_regression = keras_tuner.RandomSearch(build_model_regression,
                           objective='val_loss',
                           max_trials=3,
                           directory='C:/Users/davim/keras',
                           project_name='hiper_regression',
                           overwrite=True)

tuner_regression.search(x1,y1,epochs=3,validation_data=(x2,y2))
best_model_regression=tuner_regression.get_best_models()[0]
best_model_regression.summary()   

##Treinando o modelo com os melhores hiperparâmetros
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('lstm_model_r.h5', monitor='val_loss')
history_r=best_model_regression.fit(x1,y1,validation_data=(x2,y2),epochs=10,batch_size=200,
                                  callbacks=[early_stopping,model_checkpoint])

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(history_r.history['loss'], label='Loss do treino')
plt.plot(history_r.history['val_loss'], label='Loss da validação')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history_r.history['mean_absolute_error'], label='MAE do treino')
plt.plot(history_r.history['val_mean_absolute_error'], label='MAE do validação')
plt.ylabel('Erro Médio Absoluto (MAE)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.suptitle('Avaliação do treinamento')
plt.tight_layout()
plt.show()

##Testando o modelo com os dados de teste 
rul_prev_r = best_model_regression.predict(x_test).reshape(-1)
rul_prev_r = [float(x) for x in rul_prev_r]
end=time.time()

##Avaliando o modelo
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error
print(mean_squared_error(y_test_r,rul_prev_r))
print(f'{np.sqrt(mean_squared_error(y_test_r,rul_prev_r)):.2f}')
print(mean_absolute_percentage_error(y_test_r,rul_prev_r))
print(mean_absolute_error(y_test_r,rul_prev_r))
print(r2_score(y_test_r,rul_prev_r))  
print(f'Tempo de {end-start}s') 

n_amostras = 120
indices = np.random.choice(len(rul_prev_r),size=n_amostras,replace=False)
y_real_amostra_r=np.array(y_test_r)[indices]
y_prev_amostra_r=np.array(rul_prev_r)[indices]
ordem = np.argsort(y_real_amostra_r)
y_real_amostra_r=y_real_amostra_r[ordem]
y_prev_amostra_r=y_prev_amostra_r[ordem]

plt.figure(figsize=(10,6))
plt.scatter(range(len(indices)),y_prev_amostra_r,c='blue',label='RUL previsto')
plt.scatter(range(len(indices)),y_real_amostra_r,c='red',label='RUL real')
plt.title(f'RUL real vs RUL previsto ({n_amostras} amostras)')
plt.ylabel('RUL')
plt.xlabel('Index')
plt.legend(loc='lower right')

for i in range(len(indices)):
    plt.plot([i,i],[y_prev_amostra_r[i],y_real_amostra_r[i]],ls='--',c='black',alpha=0.7)
 

p = max(y_real_amostra_r)

plt.figure(figsize=(10,6))
plt.scatter(y_real_amostra_r,y_prev_amostra_r,c='blue')
plt.plot([0,p],[0,p],ls='--',c='red',alpha=0.7)
plt.title(f'RUL real vs RUL previsto ({n_amostras} amostras)')
plt.ylabel('RUL previsto')
plt.xlabel('RUL real')

plt.show() 

##Criação modelo classificação
shape2 = (x3.shape[1], x3.shape[2])

def build_model_classification(hp):
    model=Sequential()
    n_layers_lstm_c=hp.Int('layers_lstm_c',1,4)
    n_layers_dense_c=hp.Int('layers_dense_c',1,3)
    model.add(LSTM(hp.Int('lstm_c_0',min_value=32,max_value=256,step=32),
                   activation='tanh',
                   return_sequences=True,input_shape=shape2))
    for i in range(n_layers_lstm_c):
        return_seq = i <n_layers_lstm_c -1
        model.add(LSTM(hp.Int(f'lstm_c_{i}',min_value=32,max_value=256,step=32),
                       activation='tanh',
                       return_sequences=return_seq))
    if hp.Boolean("dropout"):
        model.add(Dropout(0.25))
    for i in range(n_layers_dense_c):
        model.add(Dense(hp.Int(f'dense_c_{i+1}',min_value=32,max_value=256,step=32),
                        activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate_c',[0.01,0.001,0.0001])), 
                  loss='binary_crossentropy', 
              metrics=['accuracy','AUC'])
    return model 

start=time.time()
tuner_classification = keras_tuner.RandomSearch(build_model_classification,
                           objective='val_loss',
                           max_trials=10,
                           directory='C:/Users/davim/keras',
                           project_name='hiper_classificassion',
                           overwrite=True)

tuner_classification.search(x3,y3,epochs=10,validation_data=(x4,y4))
best_model_classification=tuner_classification.get_best_models()[0]
best_model_classification.summary()

##Treinando o modelo com os hiperparâmetros escolhidos
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('lstm_model_c.h5', monitor='val_loss')
history_c=best_model_classification.fit(x3,y3,validation_data=(x4,y4),
                                                     epochs=100,batch_size=200,
                                                     callbacks=[early_stopping,model_checkpoint]) 

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(history_c.history['loss'], label='Loss do treino')
plt.plot(history_c.history['val_loss'], label='Loss da validação')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history_c.history['accuracy'], label='Acurácia do treino')
plt.plot(history_c.history['val_accuracy'], label='Acurácia da validação')
plt.xlabel('Epoch')
plt.ylabel('Acurácia')
plt.legend(loc='upper right')

plt.suptitle('Avaliação do treinamento de classificação')
plt.tight_layout()
plt.show()

##Teste do modelo 
rul_prev_c = best_model_classification.predict(x_test).reshape(-1)
end=time.time()
rul_prev_c_l =np.where(rul_prev_c<0.5,0,1)

##Avaliação do modelo
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,auc,roc_curve
print(accuracy_score(y_test_c,rul_prev_c_l))
print(precision_score(y_test_c,rul_prev_c_l))
print(recall_score(y_test_c,rul_prev_c_l))
print(f1_score(y_test_c,rul_prev_c_l))
print(f'Tempo de {end-start}s')

matriz=confusion_matrix(y_test_c,rul_prev_c_l)
plt.figure(figsize=(10,6))
sns.heatmap(matriz,annot=True,cmap='Blues',fmt='d',
            xticklabels=['Falha','Não Falha'],yticklabels=['Falha','Não Falha'])
plt.title('Matriz de Confusão')
plt.show()

fpr,tpr,thresholds=roc_curve(y_test_c,rul_prev_c)
auc_model = auc(fpr,tpr)

plt.figure(figsize=(10,6))
plt.plot(fpr,tpr,color='blue',label=f'ROC (Área={auc_model:2f})')
plt.plot([0,1],[0,1],ls='--',c='black',alpha=0.7)
plt.xlim(-0.05,1)
plt.ylim(0,1.05)
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

