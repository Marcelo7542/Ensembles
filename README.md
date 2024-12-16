English Description below:

Projeto de Classificação com Técnicas Ensemble

Este repositório contém implementações de duas técnicas de ensemble de aprendizado de máquina: VotingClassifier e Bagging/Pasting.

O objetivo deste projeto é entender o funcionamento dessas técnicas, comparando seus desempenhos e analisando o impacto que elas têm nos resultados de classificação.

Descrição do Projeto

Neste projeto, utilizei o dataset Olivetti Faces, que contém imagens de rostos para classificação, com o objetivo de prever a identidade das pessoas nas imagens. 
A ideia é utilizar diferentes classificadores e combiná-los em uma técnica de ensemble para melhorar a precisão do modelo.

Modelos e Técnicas Usadas

Modelos de Classificação:

Gaussian Naive Bayes (GNB): 

Modelo probabilístico baseado no teorema de Bayes, assumindo independência entre as características.

Support Vector Classifier (SVC): 

Modelo de classificação baseado em margens máximas entre as classes, utilizando um hiperplano para separar os dados.

Random Forest Classifier (RFC): 

Modelo de ensemble que utiliza múltiplas árvores de decisão para melhorar a precisão e robustez do modelo final.

Logistic Regression (LR): 

Modelo de regressão usado para prever a probabilidade de uma classe, aplicando a função sigmoide para classificação binária ou multiclasses.

Técnicas de Ensemble:

VotingClassifier:

Hard Voting: 

Previsão baseada na votação majoritária dos classificadores.

Soft Voting: 

Previsão baseada na média das probabilidades de cada classe dos classificadores.

Bagging (Bootstrap Aggregating):

Bagging com Amostragem Bootstrap: 

A técnica de Bagging treina múltiplos modelos em subconjuntos aleatórios dos dados (com reposição) e combina suas previsões.

Configuração: 

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=130, random_state=60)

Resultado: 

Acurácia: 0.89, Precisão: 0.92, Recall: 0.89, F1-Score: 0.89.

Bagging com Out-of-Bag (OOB): 

Usa amostras OOB para estimar o desempenho do modelo, sem a necessidade de um conjunto de validação separado.

Configuração: 

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=130, random_state=60, oob_score=True)

Resultado OOB: 

0.7766666666666666.

Pasting:

Utiliza todo o conjunto de treinamento para cada modelo base, sem amostragem bootstrap (sem reposição).

Configuração: 

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=130, random_state=60, bootstrap=False)

Resultado: 

Acurácia: 0.87, Precisão: 0.91, Recall: 0.87, F1-Score: 0.87.

Random Patches: 

Amostra tanto as instâncias quanto as características, aumentando a diversidade, utilizando subconjuntos aleatórios de ambos.

Configuração: 

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, max_features=0.8, bootstrap=True, bootstrap_features=True, random_state=60)

Resultado: 

Acurácia: 0.89, Precisão: 0.91, Recall: 0.89, F1-Score: 0.89.

Random Subspaces: 

Usa todas as instâncias, mas amostra aleatoriamente as características para treinar cada modelo base.

Configuração: 

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=1.0, max_features=0.8, bootstrap=False, bootstrap_features=True, random_state=60)
Resultado: Acurácia: 0.80, Precisão: 0.82, Recall: 0.80, F1-Score: 0.79.

Conclusão
O objetivo deste projeto é entender como as técnicas de ensemble funcionam e como diferentes combinações de modelos podem influenciar os resultados. 
Ao combinar classificadores com o VotingClassifier ou utilizar técnicas como Bagging e Pasting, eu pude aumentar a robustez e a precisão dos modelos.





Classification Project with Ensemble Techniques

This repository contains implementations of two machine learning ensemble techniques: 

VotingClassifier and Bagging/Pasting.

The goal of this project is to understand how these techniques work, compare their performances, and analyze the impact they have on classification results.

Project Description

In this project, I used the Olivetti Faces dataset, which contains images of faces for classification, with the goal of predicting the identity of the people in the images. 
The idea is to use different classifiers and combine them into an ensemble technique to improve the model's accuracy.

Models and Techniques Used

Classification Models:

Gaussian Naive Bayes (GNB):

A probabilistic model based on Bayes' theorem, assuming independence between features.

Support Vector Classifier (SVC):

A classification model based on maximizing margins between classes, using a hyperplane to separate the data.

Random Forest Classifier (RFC):

An ensemble model that uses multiple decision trees to improve the accuracy and robustness of the final model.

Logistic Regression (LR):

A regression model used to predict the probability of a class, applying the sigmoid function for binary or multiclass classification.

Ensemble Techniques:

VotingClassifier:

Hard Voting:

Prediction based on the majority vote of the classifiers.

Soft Voting:

Prediction based on the average of each classifier's class probabilities.

Bagging (Bootstrap Aggregating):

Bagging with Bootstrap Sampling:

The Bagging technique trains multiple models on random subsets of the data (with replacement) and combines their predictions.

Configuration:

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=130, random_state=60)

Result:

Accuracy: 0.89, Precision: 0.92, Recall: 0.89, F1-Score: 0.89.

Bagging with Out-of-Bag (OOB):

Uses OOB samples to estimate the model's performance without needing a separate validation set.

Configuration:

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=130, random_state=60, oob_score=True)

OOB Result:

0.7766666666666666.

Pasting:

Uses the entire training set for each base model, without bootstrap sampling (no replacement).

Configuration:

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=130, random_state=60, bootstrap=False)

Result:

Accuracy: 0.87, Precision: 0.91, Recall: 0.87, F1-Score: 0.87.

Random Patches:

Samples both instances and features, increasing diversity by using random subsets of both.

Configuration:

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, max_features=0.8, bootstrap=True, bootstrap_features=True, random_state=60)

Result:

Accuracy: 0.89, Precision: 0.91, Recall: 0.89, F1-Score: 0.89.

Random Subspaces:

Uses all instances but randomly samples the features to train each base model.

Configuration:

BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=1.0, max_features=0.8, bootstrap=False, bootstrap_features=True, random_state=60)

Result:

Accuracy: 0.80, Precision: 0.82, Recall: 0.80, F1-Score: 0.79.

Conclusion 

The goal of this project was to understand how ensemble techniques work and how different combinations of models can influence the results. 
By combining classifiers with VotingClassifier or using techniques like Bagging and Pasting, I could increase the robustness and accuracy of the models.
