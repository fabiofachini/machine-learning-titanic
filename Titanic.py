import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DADOS DE TREINO

# importar dados de treino e visualizar Head
df_train = pd.read_csv('train.csv')
df_train.head()

# Descrição estatística dos dados de treino
df_train.describe()

# Encontrando valores ausentes
df_train.isnull().sum()

# Identificar cidades de embarque para substituir os 2 valores null de cidades
print("Cherbourg: ",df_train['Embarked'].value_counts()['C']/len(df_train)*100,'%')
print("Queenstown: ",df_train['Embarked'].value_counts()['Q']/len(df_train)*100,'%')
print("Southampton: ",df_train['Embarked'].value_counts()['S']/len(df_train)*100,'%')

# Identificando as linhas com valores ausentes de cidade de embarque
df_train[df_train['Embarked'].isnull()]

# Substituindo valores ausentes de cidade de embarque pela maior ocorrência
df_train['Embarked'].fillna('S', inplace = True)

# Criar gráfico de pizza de cidades de embarque
pizza_embarked = {}
for item in df_train['Embarked']:
    if item in pizza_embarked:
        pizza_embarked[item] += 1
    else:
        pizza_embarked[item] = 1

labels_embarked = list(pizza_embarked.keys())
valores_embarked = list(pizza_embarked.values())

plt.pie(valores_embarked, labels=labels_embarked, autopct='%1.1f%%')
plt.title('Cidades de Embarque')
plt.show()

# Criar gráfico de pizza de sexo
Sexo = df_train.loc[:,'Sex']
pizza_Sexo = {}
for item in Sexo:
    if item in pizza_Sexo:
        pizza_Sexo[item] += 1
    else:
        pizza_Sexo[item] = 1

labels_Sexo = list(pizza_Sexo.keys())
valores_Sexo = list(pizza_Sexo.values())

plt.pie(valores_Sexo, labels=labels_Sexo, autopct='%1.1f%%', colors = ['b','m'])
plt.title('Sexo')
plt.show()

# Criar gráfico de pizza de sexo que sobreviveu
Sexo_Survived = df_train.loc[df_train['Survived'] == 1,'Sex']
pizza_Sexo_Survived = {}
for item in Sexo_Survived:
    if item in pizza_Sexo_Survived:
        pizza_Sexo_Survived[item] += 1
    else:
        pizza_Sexo_Survived[item] = 1

labels_Sexo_Survived = list(pizza_Sexo_Survived.keys())
valores_Sexo_Survived = list(pizza_Sexo_Survived.values())

plt.pie(valores_Sexo_Survived, labels=labels_Sexo_Survived, autopct='%1.1f%%', colors = ['m','b'])
plt.title('Sexo Que Sobreviveu')
plt.show()

# Criar gráfico de pizza de classes
Pclass = df_train.loc[:,'Pclass']
pizza_Pclass = {}
for item in Pclass:
    if item in pizza_Pclass:
        pizza_Pclass[item] += 1
    else:
        pizza_Pclass[item] = 1

labels_Pclass = list(pizza_Pclass.keys())
valores_Pclass = list(pizza_Pclass.values())

plt.pie(valores_Pclass, labels=labels_Pclass, autopct='%1.1f%%', colors = ['r','royalblue','g'])
plt.title('Classes')
plt.show()

# Criar gráfico de pizza de classes que sobreviveram
Pclass_Survived = df_train.loc[(df_train['Survived']==1),'Pclass']
pizza_Pclass_Survived = {}
for item in Pclass_Survived:
    if item in pizza_Pclass_Survived:
        pizza_Pclass_Survived[item] += 1
    else:
        pizza_Pclass_Survived[item] = 1

labels_Pclass_Survived = list(pizza_Pclass_Survived.keys())
valores_Pclass_Survived = list(pizza_Pclass_Survived.values())

plt.pie(valores_Pclass_Survived, labels=labels_Pclass_Survived, autopct='%1.1f%%', colors = ['royalblue', 'r', 'g'])
plt.title('Classes Que Sobreviveram')
plt.show()

# Criar gráfico de pizza de classes que morreram
Pclass_morte = df_train.loc[(df_train['Survived']==0),'Pclass']
pizza_Pclass_morte = {}
for item in Pclass_morte:
    if item in pizza_Pclass_morte:
        pizza_Pclass_morte[item] += 1
    else:
        pizza_Pclass_morte[item] = 1

labels_Pclass_morte = list(pizza_Pclass_morte.keys())
valores_Pclass_morte = list(pizza_Pclass_morte.values())

plt.pie(valores_Pclass_morte, labels=labels_Pclass_morte, autopct='%1.1f%%', colors = ['r', 'royalblue', 'g'])
plt.title('Classes Que Morreram')
plt.show()

# Criar histograma de idades
Age = df_train['Age']
plt.hist(Age, 20)
plt.title('Histograma de Idades')
plt.show()

# Criar histograma de tarifa
Fare = df_train['Fare']
plt.hist(Fare, 100)
plt.title('Histograma de Tarifa')
plt.show()

# Criar gráfico de pizza de Irmãos/Cônjuges
SibSp = df_train.loc[:,'SibSp']
pizza_SibSp = {}
for item in SibSp:
    if item in pizza_SibSp:
        pizza_SibSp[item] += 1
    else:
        pizza_SibSp[item] = 1

labels_SibSp = list(pizza_SibSp.keys())
valores_SibSp = list(pizza_SibSp.values())

plt.pie(valores_SibSp, labels=labels_SibSp, autopct='%1.1f%%')
plt.title('Irmãos e Cônjuges')
plt.show()

# Criar gráfico de pizza de filhos
Parch = df_train.loc[:,'Parch']
pizza_Parch = {}
for item in Parch:
    if item in pizza_Parch:
        pizza_Parch[item] += 1
    else:
        pizza_Parch[item] = 1

labels_Parch = list(pizza_Parch.keys())
valores_Parch = list(pizza_Parch.values())

plt.pie(valores_Parch, labels=labels_Parch, autopct='%1.1f%%')
plt.title('Filhos')
plt.show()

# Identificar % de mulheres da primeira classe que sobreviveram
Female_first_survived = df_train.loc[(df_train['Pclass'] == 1) & (df_train['Sex'] == 'female') & (df_train['Survived'] == 1)]
Female_first_total = df_train.loc[(df_train['Pclass'] == 1) & (df_train['Sex'] == 'female')]
Tamanho_Female_first_survived = len(Female_first_survived)
Tamanho_Female_first_total = len(Female_first_total)
print(Tamanho_Female_first_survived / Tamanho_Female_first_total)

# Identificar % de homens da primeira classe que sobreviveram
Male_first_survived = df_train.loc[(df_train['Pclass'] == 1) & (df_train['Sex'] == 'male') & (df_train['Survived'] == 1)]
Male_first_total = df_train.loc[(df_train['Pclass'] == 1) & (df_train['Sex'] == 'male')]
Tamanho_Male_first_survived = len(Male_first_survived)
Tamanho_Male_first_total = len(Male_first_total)
print(Tamanho_Male_first_survived / Tamanho_Male_first_total)

# Identificar % de mulheres da terceira classe que sobreviveram
Female_third_survived = df_train.loc[(df_train['Pclass'] == 3) & (df_train['Sex'] == 'female') & (df_train['Survived'] == 1)]
Female_third_total = df_train.loc[(df_train['Pclass'] == 3) & (df_train['Sex'] == 'female')]
Tamanho_Female_third_survived = len(Female_third_survived)
Tamanho_Female_third_total = len(Female_third_total)
print(Tamanho_Female_third_survived / Tamanho_Female_third_total)

# Identificar % de homens da terceira classe que sobreviveram
Male_third_survived = df_train.loc[(df_train['Pclass'] == 3) & (df_train['Sex'] == 'male') & (df_train['Survived'] == 1)]
Male_third_total = df_train.loc[(df_train['Pclass'] == 3) & (df_train['Sex'] == 'male')]
Tamanho_Male_third_survived = len(Male_third_survived)
Tamanho_Male_third_total = len(Male_third_total)
print(Tamanho_Male_third_survived / Tamanho_Male_third_total)

# DADOS DE TESTE

# importar dados de treino e visualizar
df_test = pd.read_csv('test.csv')
df_test.head()

# Descrição estatística dos dados de treino
df_test.describe()

# Encontrando valores ausentes
df_test.isnull().sum()

# Identificando as linhas com valores ausentes
df_test[df_test['Fare'].isnull()]

# Substituindo valores ausentes pela média
df_test['Fare'].fillna(df_test['Fare'].mean(), inplace = True)

# MACHINE LEARNING

# Treinamento dos modelos
y = df_train['Survived']
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])

# Regressão Logística
from sklearn.linear_model import LogisticRegression
reg_log = LogisticRegression(random_state=0, C = 1).fit(X, y)
pred_reg_log = reg_log.predict(X_test)
output_reg_log = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_reg_log})
output_reg_log.to_csv('submission_reg_log.csv', index=False)
print(output_reg_log)

# KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=2).fit(X, y)
pred_KNN = KNN.predict(X_test)
output_KNN = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_KNN})
output_KNN.to_csv('submission_KNN.csv', index=False)
print(output_KNN)

# SVM
from sklearn import svm
SVM = svm.SVC()
SVM.fit(X, y)
pred_SVM = SVM.predict(X_test)
output_SVM = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_SVM})
output_SVM.to_csv('submission_SVM.csv', index=False)
print(output_SVM)

# Decision Tree
from sklearn import tree
Tree = tree.DecisionTreeClassifier()
Tree.fit(X, y)
pred_Tree = Tree.predict(X_test)
output_Tree = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_Tree})
output_Tree.to_csv('submission_Tree.csv', index=False)
print(output_Tree)

# Random Forests
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=5)
RF.fit(X, y)
pred_RF = RF.predict(X_test)
output_RF = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_RF})
output_RF.to_csv('submission_RF.csv', index=False)
print(output_RF)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X, y)
pred_NB = NB.predict(X_test)
output_NB = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_NB})
output_NB.to_csv('submission_NB.csv', index=False)
print(output_NB)

# MLP
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=500)
MLP.fit(X, y)
pred_MLP = MLP.predict(X_test)
output_MLP = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_MLP})
output_MLP.to_csv('submission_MLP.csv', index=False)
print(output_MLP)