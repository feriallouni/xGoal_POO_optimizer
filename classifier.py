import os
from model.model import ModelOptimizer
from model.process import (
    DataEncoder,
    DataSplitter,
    EventProcessor
)

if __name__ == "__main__":
    print("Quel model de classification souhaitez-vous choisir ?\n1- GradientBoosting Classifier\n2- MODEL RandomForest CLASSIFIER\n3- MODEL LogisticRegression \n4- Quit")
    choix = input("Merci d'entrer votre choix : ")


    # Data processing
    events_file = os.getcwd() + "/data/events.csv"    
    info_file = os.getcwd() + "/data/ginf.csv"
    processor = EventProcessor(events_file, info_file)
    processor.process()
    encodeur = DataEncoder(processor.shots)
    encodeur.encode_categorical_variables()
    splitter = DataSplitter(
        data=encodeur.encoded_data.iloc[:,:-1],
        target=encodeur.encoded_data.iloc[:,-1]
    )
    splitter.split()

    # Utilisation des modèles Training et Test
    optimizer = ModelOptimizer(
        X_train=splitter.X_train,
        X_test=splitter.X_test,
        y_train=splitter.y_train,
        y_test=splitter.y_test
    )
    

    
    while choix != '4':
        
        if(choix == '1'):  
            # GradientBoostingClassifier
            optimizer.fmin_GBC(max_evals=10)
            best_score = optimizer.get_best_score_GBC()
            optimized_model = optimizer.choice_best_params_GBC(best_score)
            print('Voici les résultats pour le modèle Gradient Boosting Classifier :')
            optimizer.evaluate_metrics(optimized_model)

        if(choix == '2'):
            # RandomForestClassifier
            optimizer.fmin_RF(max_evals=10)
            best_score = optimizer.get_best_score_RF()
            optimized_model = optimizer.choice_best_params_RF(best_score)
            print('Voici les résultats pour le modèle RandomForest Classifier :')
            optimizer.evaluate_metrics(optimized_model)


        if(choix == '3'):
            #LogisticRegression 
            optimizer.fmin_LR(max_evals=10)
            best_score = optimizer.get_best_score_LR()
            optimized_model = optimizer.choice_best_params_LR(best_score)
            print('Voici les résultats pour le modèle LogisticRegression :')
            optimizer.evaluate_metrics(optimized_model)

        elif(choix == '4'):
            print("Merci pour votre visite")

            
        else:
            print("Invalid choice")

        print("Quel model de classification souhaitez-vous choisir ?\n1- GradientBoostingClassifier\n2- MODEL RandomForestClassifier\n3- MODEL LogisticRegression \n4- Quit")
        choix = input("Merci d'entrer votre choix : ")


    




