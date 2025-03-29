# Previsão de Vendas de Sorvetes com AzureML
Este repositório contém os arquivos de código e outros recursos necessários para construir um modelo de regressão preditiva com Machine Learning. O objetivo é prever as vendas de sorvete da sorveteria "Gelato Mágico" com base na temperatura ambiente. Este projeto foi desenvolvido como parte de um desafio prático da DIO para demonstrar habilidades em criar e implementar modelos de Machine Learning para problemas de previsão de demanda, especificamente, um modelo de regressão.

# Passo 1 - Obtenção da base de dados 
A base de dados foi foi obtida na platforam kaggle, [Ice Cream Sales - temperatures](https://www.kaggle.com/datasets/raphaelmanayon/temperature-and-ice-cream-sales). A base conta com colunas Temperature e Ice cream profits, que representam a temperatura ambiente e o total de vendas do dia.

# Passo 2 - Análise explorátoria dos dados
Antes de continuar, se faz necessário conhecer bem o dataset e tratar possiveis erros e implementar algumas melhorias. Para tal foi realizado o seguinte pipeline com as bibliotecas pandas e seaborn no notebook Gelato_Vendas_Regression criado no AzureML.

  - **Dimensões do dataset:**
    

```python
df.shape

```

-   **Visualização das primeiras linhas:**
    

```python
df.head()

```

-   **Informações sobre os tipos de dados:**
    

```python
df.info()

```

-   **Estatísticas descritivas:**
    

```python
df.describe()

```

Visualizações para entender a distribuição e relação entre variáveis:

-   **Boxenplot para "Ice Cream Profits"**
    

```python
sns.boxenplot(data=df['Ice Cream Profits'])

```

-   **Boxenplot para "Temperature"**
    

```python
sns.boxenplot(data=df['Temperature'])

```

-   **Pairplot para visualizar relações entre as variáveis**
    

```python
sns.pairplot(df)

```

-   **Matriz de correlação**
    

```python
df.corr()

```

Após as análises, observou-se que o dataset não possuia outliers, dados faltantes ou qualquer tipo de irregulariade. Além de que a váriavel target (Ice cream profits) apresenta comportamento linear e forte correlação com a temperatura.
![Image](https://github.com/user-attachments/assets/ebcc5e1f-140c-46fc-89f5-2877f560a66b)

## Passo 3. Pré-processamento dos Dados
**Ativação do MLflow para monitoramento e gerenciamento do modelo a ser criado:**

```python
# Definir o nome

mlflow.set_experiment("Gelato-Regression_nt") 

# Ativar o autologging com MLflow

mlflow.sklearn.autolog()

```


-   **Separação entre features e target:**
    

```python
X = df.drop('Ice Cream Profits', axis=1)
y = df['Ice Cream Profits']

```

-   **Padronização dos dados:**
    

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)

```

-   **Divisão entre treino e teste:**
    

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

## Passo 4. Treinamento dos Modelos
Como os dados apresentam comportamento linear, implimentamos dois algoritimos que possuem como base essa função. Sendo eles:
### 6.1 Regressão Linear

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

```

### 6.2 Regressão Ridge

```python
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge.score(X_test, y_test)

```

----------
## Passo 5. Avaliação do Resultados
Entre os dois algorimos implementados, o LinearRegression apresentou o melhor desempenho 0.9736 de r2_score na predição do volume de vendas com base na variação da temperatura. Dessa forma, tornou-se o algoritmo escolhido para a implementação do modelo final. 

## Passo 6. Criação do script e componente para o pipeline.

Para tal, foi implementado as seguintes modalidades para a criação do modelo. A primeira foi por comando, encapsulando o modelo desenvolvido no notebook [Gelato_Vendas_Regression.ipynb](https://github.com/BrendoEnomoto/Previsao-de-Vendas-de-Sorvetes-com-Machine-Learning/blob/main/Gelato_Vendas_Regression.ipynb 
"Gelato_Vendas_Regression.ipynb")  através de outro notebook presente no respositório
[galato_regression_model.ipynb](https://github.com/BrendoEnomoto/Previsao-de-Vendas-de-Sorvetes-com-Machine-Learning/blob/main/galato_regression_model.ipynb "galato_regression_model.ipynb"). Por outro lado, também foi implementado o mesmo pipeline no AzureML Designer no qual apresentou os seguintes resultados:
![Image](https://github.com/user-attachments/assets/5b91af7b-3e33-4952-8255-837fd73e71b9)
![Image](https://github.com/user-attachments/assets/be31d484-9613-4e22-a61c-bb3a5561789a)


## Passo 7 - Implentação do modelo em ponto de extremidade em tempo real.
Para finaliizar, após encapsular o modelo desenvolvido nos notebooks e salvar como ativo (modelo), este foi disponibiilizado em um ponto de extremidade em tempo real, como demonstrado na imagem abaixo:
![Image](https://github.com/user-attachments/assets/0cd299ea-8586-49ec-9d3d-7e1b06c5deb2)

