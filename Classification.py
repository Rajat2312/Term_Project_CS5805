# -----------------------------------------------------------
#               Phase 3: Classification Analysis
# -----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import time
from itertools import cycle, combinations
from prettytable import PrettyTable
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings(action='ignore')


result_table = PrettyTable(field_names=['Classifier', 'Confusion_Matrix', 'Macro-Precision', 'Macro-Recall', 'Macro-F1', 'Macro-Specificity'
                                        , 'Macro ROC-AUC(OvR)', 'Macro ROC-AUC(OvO)'])
result_table.title = 'Evaluation of various classifiers'


def evaluate_model(model, y_true, y_pred, y_proba, model_name):
    """
    :param model:  Classifier to evaluate
    :param y_true: target class labels
    :param y_pred: predicted class labels
    :param model_name: Classifier name in string (passed as string in method call)
    """
    print('-------------- Evaluating for model {} --------------'.format(str(model_name)))
    # 1. Confusion Matrix
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    print('Confusion Matrix for {} is:'.format(model_name))
    print(cnf_matrix)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title('{} Confusion matrix'.format(model_name), y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    # 2. Precision (Macro and micro)
    mac_precision = metrics.precision_score(y_true, y_pred, average='macro')
    mic_precision = metrics.precision_score(y_true, y_pred, average='micro')
    print('Macro Precision for {} is: {}'.format(str(model), round(mac_precision, 3)))
    print('Micro Precision for {} is: {}'.format(str(model), round(mic_precision, 3)))
    print()
    # 3. Recall (Macro & Micro)
    mac_recall = metrics.recall_score(y_true, y_pred, average='macro')
    mic_recall = metrics.recall_score(y_true, y_pred, average='micro')
    print('Macro Recall for {} is: {}'.format(str(model), round(mac_recall, 3)))
    print('Micro Recall for {} is: {}'.format(str(model), round(mic_recall, 3)))
    print()
    # 4. F1 Score (Macro & Micro)
    mac_f1 = metrics.recall_score(y_true, y_pred, average='macro')
    mic_f1 = metrics.recall_score(y_true, y_pred, average='micro')
    print('Macro F1 for {} is: {}'.format(str(model), round(mac_f1, 3)))
    print('Micro F1 for {} is: {}'.format(str(model), round(mic_f1, 3)))
    print()

    # 5. Specificity (TN / TN + FP)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # labels=[0,1,2]
    # for i in labels:
    # Specificity
    #     label_specificities = [specificity(y_true[:, i].ravel(), y_pred[:, i].ravel()) for i in range(y_true.shape[i])]
    #     # Calculate micro specificity
    #     micro_specificity = np.sum(label_specificities) / len(label_specificities)
    #     # Calculate macro specificity
    #     macro_specificity = np.mean(label_specificities)
    #     print('Micro Specificity for model {} is: {}'.format(str(model), micro_specificity))
    #     print('Macro Specificity for model {} is: {}'.format(str(model), macro_specificity))
    # print(f"Macro Specificity: {macro_specificity}")
    specificity = TN / (TN + FP)
    mac_specificity = np.mean(specificity)
    print('Macro-specificity for {} is: {}'.format(model_name, round(mac_specificity,3)))
    # print('Specificity for class 0:', round(specificity[0], 3))
    # print('Specificity for class 1:', round(specificity[1], 3))
    # print('Specificity for Class 2:', round(specificity[2], 3))

    # 6 Macro and Micro ROC_AUC score (OvR)
    macro_auc_ovr = metrics.roc_auc_score(y_true, y_proba, labels=model.classes_, multi_class='ovr', average='macro')
    micro_auc_ovr = metrics.roc_auc_score(y_true, y_proba, labels=model.classes_, multi_class='ovr', average='micro')
    print('Macro-averaged One v/s Rest ROC AUC score: {}'.format(round(macro_auc_ovr, 3)))
    print('Micro-averaged One v/s Rest ROC AUC score: {}'.format(round(micro_auc_ovr, 3)))

    # # 7. Macro and Micro ROC_AUC score (OvO)
    macro_auc_ovo = metrics.roc_auc_score(y_true, y_proba, labels=model.classes_, multi_class='ovo', average='macro')
    # micro_auc_ovo = metrics.roc_auc_score(y_true, y_proba, labels=model.classes_, multi_class='ovo', average='micro')
    print('Macro AUC (OvO) for model {} is: {}'.format(model_name, round(macro_auc_ovo, 3)))
    # print('Micro AUC for(OvO) model {} is: {}'.format(str(model), round(micro_auc_ovo, 3)))

    # Append result to table
    # field_names=['Classifier', 'Confusion_Matrix', 'Macro-Precision', 'Macro-Recall', 'Macro-F1', 'Macro-Specificity'
    #                                         , 'Macro ROC-AUC(OvR)', 'Macro ROC-AUC(OvO)'])
    result_table.add_row([model_name, cnf_matrix, round(mac_precision,3), round(mac_recall,3), round(mac_f1,3),
                          round(mac_specificity,3), round(macro_auc_ovr,3),
                          round(macro_auc_ovo,3)])


def plot_roc_ovr(y_train, y_test, y_proba, model_name):
    # We need macro and micro averages of roc_auc, along with class-wise roc_auc
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    # Compute micro AUC-ROC
    fpr, tpr, roc_auc = {}, {}, {}
    fpr['micro'], tpr['micro'], _ = metrics.roc_curve(y_onehot_test.ravel(), y_proba.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro AUC-ROC
    n_classes = len(np.unique(y_train))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_onehot_test[:, i], y_proba[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0,1.0,1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])   # Linear Interpolation

    # Take average and compute AUC
    mean_tpr /= n_classes
    fpr['macro'] = fpr_grid
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = metrics.auc(fpr['macro'], tpr['macro'])

    # Plot all OvR ROC curves together
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr['micro'], tpr['micro'],
        label=f"Micro average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color='deeppink',
        linestyle=':',
        linewidth=4)
    plt.plot(
        fpr['macro'], tpr['macro'],
        label=f"Macro average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color='navy',
        linestyle=':',
        linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_proba[:, class_id],
            name='ROC Curve for Class {}'.format(class_id),
            color=color,
            ax=ax,
            plot_chance_level=(class_id==2),
        )
    plt.axis('square')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Extension of ROC for {} to \nOne-vs-Rest multiclass'.format(model_name))
    plt.legend()
    plt.show()


def plot_roc_ovo(y_train, y_test, y_proba, model_name):
    label_binarizer = LabelBinarizer().fit(y_train)
    pair_list = list(combinations(np.unique(y_train), 2))
    pair_scores = []
    mean_tpr = dict()
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    for ix, (label_a, label_b) in enumerate(pair_list):
        a_mask = y_test == label_a
        b_mask = y_test == label_b
        ab_mask = np.logical_or(a_mask, b_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]
        idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]

        fpr_a, tpr_a, _ = metrics.roc_curve(a_true, y_proba[ab_mask, idx_a])
        fpr_b, tpr_b, _ = metrics.roc_curve(b_true, y_proba[ab_mask, idx_b])

        mean_tpr[ix] = np.zeros_like(fpr_grid)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr[ix] /= 2
        mean_score = metrics.auc(fpr_grid, mean_tpr[ix])
        pair_scores.append(mean_score)

    ovo_tpr = np.zeros_like(fpr_grid)
    macro_roc_auc_ovo = metrics.roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovo",
        average="macro",
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    for ix, (label_a, label_b) in enumerate(pair_list):
        ovo_tpr += mean_tpr[ix]
        plt.plot(
            fpr_grid,
            mean_tpr[ix],
            label=f"Mean {label_a} vs {label_b} (AUC = {pair_scores[ix]:.3f})",
        )

    ovo_tpr /= sum(1 for pair in enumerate(pair_list))

    plt.plot(
        fpr_grid,
        ovo_tpr,
        label=f"One-vs-One macro-average (AUC = {macro_roc_auc_ovo:.3f})",
        linestyle=":",
        linewidth=4,
    )
    plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Extension of ROC for {model_name} to \nOne-vs-One multiclass")
    plt.legend()
    plt.show()


def stratified_kfold_cv(X, y, model, model_name):
    print(f'---------- Stratified K-Fold CV for {model_name} ----------')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
    prec, rec, f1 = [], [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        if 'KNN' in model_name:
            # Fit the model
            model.fit(X_train.values, y_train.values)
            # Predict the credit score
            y_pred = model.predict(X_test.values)
            y_pred_proba = model.predict_proba(X_test.values)
        else:
            # Fit the model
            model.fit(X_train, y_train)
            # Predict the credit score
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        # Evaluate model
        prec += [metrics.precision_score(y_test, y_pred, average='macro')]
        rec += [metrics.recall_score(y_test, y_pred, average='macro')]
        f1 += [metrics.f1_score(y_test, y_pred, average='macro')]
        # auc += [metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr',average='macro')]
    print('------------------Stratified K-Fold CV Results------------------')
    print('Macro averaged Precision Score for {} = {}'.format(model_name, round(np.mean(prec), 3)))
    print('Macro averaged Recall Score for {} = {}'.format(model_name, round(np.mean(rec), 3)))
    print('Macro averaged F1 Score for {} = {}'.format(model_name, round(np.mean(f1), 3)))
    # print('Macro averaged ROC-AUC score for {} = {}'.format(model_name))


if __name__ == "__main__":
    start = time.time()
    # ------------------------------
    #       Read Cleaned Data
    # ------------------------------
    df = pd.read_csv('credit_classification_cleaned.csv')
    print(df.info())
    print(df.head())
    print(df.columns)
    print('Shape:', df.shape)
    # Drop additional columns
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Split into X & y
    X = df.drop('Credit_Score', axis=1)
    y = df['Credit_Score']
    # Downsample the data
    down_sample = RandomUnderSampler(sampling_strategy={0: 4000, 1: 4000, 2: 4000}, random_state=5805)
    X, y = down_sample.fit_resample(X, y)
    # ---------------------------------
    #       Train Test Split
    # ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

    # =============================================================================
    #                   ONE V/S ALL OR ONE V/S REST CLASSIFICATION
    # =============================================================================
    # -----------------------------------------------------------
    #           Classifier 1: Decision Tree
    # -----------------------------------------------------------
    print('-----------------Simple Decision Tree------------------')
    dt_model = DecisionTreeClassifier(random_state=5805)
    dt_model.fit(X_train, y_train)
    # # #
    # # # # Predict credit_score
    y_pred_train = dt_model.predict(X_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_dt_proba = dt_model.predict_proba(X_test)
    # # # # #
    # # # # # Show metrics like accuracy of test and train set to check overfitting
    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_dt)
    print('Accuracy of train:', round(train_acc, ))
    print('Accuracy of test:', round(test_acc, 2))
    evaluate_model(dt_model, y_test, y_pred_dt, y_pred_dt_proba, model_name='Decision Tree')
    plot_roc_ovr(y_train, y_test,y_pred_dt_proba, model_name='Decision Tree')
    plot_roc_ovo(y_train, y_test, y_pred_dt_proba, model_name='Decision Tree')
    stratified_kfold_cv(X, y, dt_model, 'Decision Tree')
    # # # #
    # # # # # Grid Search CV for parameters of decision tree classifier
    # # start_time = time.time()
    # parameters = {'max_depth': [i for i in range(5, 16, 5)],
    #               'min_samples_split': [i for i in range(5, 16, 5)],
    #               'min_samples_leaf': [i for i in range(5, 16, 5)],
    #               'max_features': [5, 10, 15, 'sqrt','log2'],
    #               'splitter': ['best'],
    #               'criterion': ['gini', 'entropy']}
    # grid_model = GridSearchCV(estimator=dt_model, param_grid=parameters, scoring='accuracy',n_jobs=-1)
    # grid_model.fit(X_train, y_train)
    print('After Grid Search best parameters are:')
    # print(grid_model.best_params_)
    # best_parameters = grid_model.best_params_
    best_parameters = {'criterion': 'gini', 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'min_samples_split': 15, 'splitter': 'best'}
    print('Best Parameters Decision Tree:', best_parameters)
    # # # #     # grid_model.best_params_)
    # end_time = time.time()
    # # time_spent = end_time - start_time
    # # print('Time (mins) for Grid Search:', round(time_spent/60, 2))
    print('-----------------Decision Tree Pre-Pruned------------------')
    dt_model_pre = DecisionTreeClassifier(**best_parameters, random_state=5805)
    dt_model_pre.fit(X_train, y_train)
    y_pred_dt = dt_model_pre.predict(X_test)
    y_pred_proba_dt = dt_model_pre.predict_proba(X_test)
    # # # # # Evaluate Model
    evaluate_model(dt_model_pre, y_test, y_pred_dt, y_pred_proba_dt, model_name='Pre-pruned Decision Tree')
    plot_roc_ovr(y_train, y_test, y_pred_proba_dt, model_name='Pre-pruned Decision Tree')
    plot_roc_ovo(y_train, y_test, y_pred_proba_dt, model_name='Pre-pruned Decision Tree')
    stratified_kfold_cv(X, y, dt_model_pre, 'Pre-pruned Decision Tree')
    #
    # # # # ----------------------------------------------------------
    # # # #           Post-Pruning (Optimum alpha)
    # # # # ----------------------------------------------------------
    path = dt_model.cost_complexity_pruning_path(X_train, y_train)
    alphas = path['ccp_alphas']
    # print(alphas)
    print('No.of alphas:',len(alphas))
    # print(alphas)
    # # =============================
    # # Grid search for best alpha
    # # =============================
    accuracy_train, accuracy_test = [], []
    for i in range(0, len(alphas), 200):
        dt_model = DecisionTreeClassifier(ccp_alpha=alphas[i], random_state=5805)
        dt_model.fit(X_train, y_train)
        y_train_pred = dt_model.predict(X_train)
        y_test_pred = dt_model.predict(X_test)
        accuracy_train.append(metrics.accuracy_score(y_train, y_train_pred))
        accuracy_test.append(metrics.accuracy_score(y_test, y_test_pred))
    alpha_range = [alphas[i] for i in range(0,len(alphas),200)]
    fig, ax = plt.subplots()
    ax.set_xlabel('alpha')
    ax.set_ylabel('accuracy')
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(alpha_range, accuracy_train, marker="o", label="train",
            drawstyle="steps-post")
    ax.plot(alpha_range, accuracy_test, marker="o", label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    if X.shape[1] < 50000:
        opt_alpha = 0.00021
    else:
        opt_alpha = 0.0000325

    dt_model = DecisionTreeClassifier(ccp_alpha=opt_alpha, random_state=5805)
    dt_model.fit(X_train, y_train)
    # # # # Predict
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)
    evaluate_model(dt_model, y_test, y_pred_dt, y_pred_proba_dt, model_name='Post-Pruned Decision Tree')
    plot_roc_ovr(y_train, y_test, y_pred_proba_dt, model_name='Post-pruned Decision Tree')
    plot_roc_ovo(y_train, y_test, y_pred_proba_dt, model_name='Post-pruned Decision Tree')
    stratified_kfold_cv(X, y, dt_model, 'Post-pruned Decision Tree')
    # # # Experiments: 100-200, 200-300, 900-1000,1500-1600, 3000-3100,5000-5200
    # #
    # # # # -------------------------------------------------------------------------
    # # # #           Classifier 2: Logistic Regression
    # # # # -------------------------------------------------------------------------
    print('-----------------Logistic Regression------------------')
    multi_logistic = LogisticRegression(C=0.5, multi_class='ovr', random_state=5805)  # OvR Classifier
    multi_logistic.fit(X_train, y_train)
    # #
    # # # Make Prediction
    y_pred_log = multi_logistic.predict(X_test)
    y_pred_proba_log = multi_logistic.predict_proba(X_test)
    # # # # # Evaluate OvR
    evaluate_model(multi_logistic, y_test, y_pred_log, y_pred_proba_log, model_name='Logistic Regression')
    plot_roc_ovr(y_train, y_test, y_pred_proba_log, model_name='Logistic Regression')
    plot_roc_ovo(y_train,y_test,y_pred_proba_log, model_name='Logistic Regression')
    stratified_kfold_cv(X, y, multi_logistic, 'Logistic Regression')
    #
    # # # ------------Grid Search CV for Logistic Regression----------------
    parameters = {'C': [1.0, 2.0],
                  'solver': ['lbfgs', 'liblinear']}
    grid_log = GridSearchCV(estimator=multi_logistic, param_grid=parameters, scoring='f1_macro')
    grid_log.fit(X_train, y_train)
    # best_parameters = grid_log.best_params_
    best_parameters = {'C': 2.0, 'solver': 'lbfgs'}
    print('Best Parameters for Logistic Regression:', best_parameters)
    grid_log = LogisticRegression(**best_parameters,random_state=5805)
    grid_log.fit(X_train, y_train)
    # #
    y_pred_log = grid_log.predict(X_test)
    y_pred_proba_log = grid_log.predict_proba(X_test)
    # # Evaluate the model
    evaluate_model(grid_log, y_test, y_pred_log, y_pred_proba_log, model_name='Grid Logistic Regression')
    plot_roc_ovr(y_train, y_test, y_pred_proba_log, model_name='Grid Logistic Regression')
    plot_roc_ovo(y_train, y_test, y_pred_proba_log, model_name='Grid Logistic Regression')
    stratified_kfold_cv(X, y, grid_log, 'Grid Logistic Regression')
    #
    #
    # #
    # # # -------------------------------------------------------------------------
    # # #           Classifier 3: K-Nearest Neighbors(KNN)
    # # # -------------------------------------------------------------------------
    print('-----------------K-Nearest Neighbors(KNN)------------------')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train.values, y_train.values)
    y_pred_knn = knn.predict(X_test.values)
    y_pred_knn_proba = knn.predict_proba(X_test.values)
    print('--------------Results before Optimal K--------------')
    evaluate_model(knn, y_test, y_pred_knn, y_pred_knn_proba, model_name='KNN')
    plot_roc_ovr(y_train, y_test, y_pred_knn_proba, model_name='KNN')
    plot_roc_ovo(y_train, y_test, y_pred_knn_proba, model_name='KNN')
    stratified_kfold_cv(X, y, knn, 'KNN')
    #
    #
    ## Select Best K
    print('----------Grid Search for KNN Classifier----------')
    k_options = [i for i in range(1, 12)]
    error_rate = []
    for k in k_options:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train.values,y_train.values)
        y_pred = knn.predict(X_test.values)
        # Get average error per record for each knn model
        error_rate.append(np.mean(y_test.values != y_pred))
    # ## Plot Error Rate vs K
    plt.figure(figsize=(10, 6))
    # k_range = [i for i in range(1,21,1)]
    plt.plot(k_options, error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',markersize=12)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.xticks(ticks=k_options)
    plt.show()
    knn = KNeighborsClassifier()
    from sklearn.model_selection import GridSearchCV

    # k_range = list(range(3, 11))
    # param_grid = dict(n_neighbors=k_range)
    # grid = GridSearchCV(knn, param_grid, cv=2, scoring='f1_macro',
    #                     return_train_score=True, verbose=1)
    # # fitting the model for grid search
    # grid_search = grid.fit(X_train.values, y_train.values)
    # print(f'The best k after grid search is : {grid_search.best_params_}')
    # #
    print('------------------Results after selecting Optimal K------------------')
    optimal_k = 9  # Used Elbow Method to select K
    print('Optimal K:', optimal_k)
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train.values, y_train.values)
    y_pred_knn = knn.predict(X_test.values)
    y_pred_knn_proba = knn.predict_proba(X_test.values)
    # # # # print(y_pred_knn_proba)
    evaluate_model(knn, y_test, y_pred_knn, y_pred_knn_proba, model_name='KNN(Optimal K)')
    plot_roc_ovr(y_train, y_test, y_pred_knn_proba, model_name='KNN(Optimal K)')
    plot_roc_ovo(y_train, y_test, y_pred_knn_proba, model_name='KNN(Optimal K)')
    stratified_kfold_cv(X, y, knn, 'KNN(Optimal K) Classifier')
    # #
    # # # # -------------------------------------------------------------------------
    # # # #           Classifier 4: Support Vector Machine(SVM)
    # # # # -------------------------------------------------------------------------
    print('-----------------Support Vector Machine(SVM)------------------')
    svc_linear = SVC(C=0.1, kernel='linear', random_state=5805, probability=True)
    svc_poly = SVC(C=0.1, kernel='poly', random_state=5805, probability=True)
    svc_rbf = SVC(C=0.1, kernel='rbf', random_state=5805,probability= True)
    # # # #
    # # # # # Fit the models
    print('-----------------Linear SVM Classifier(Fit in progress)------------------')
    svc_linear.fit(X_train, y_train)
    print('-----------------Poly SVM Classifier(Fit in progress)------------------')
    svc_poly.fit(X_train, y_train)
    print('-----------------RBF SVM Classifier(Fit in progress)------------------')
    svc_rbf.fit(X_train, y_train)
    # # # #
    # # # # # Predicting the target
    y_pred_svc_linear = svc_linear.predict(X_test)
    y_pred_svc_poly = svc_poly.predict(X_test)
    y_pred_svc_rbf = svc_rbf.predict(X_test)
    # # # #
    # # # # # Predicting the probability
    y_proba_linear = svc_linear.predict_proba(X_test)
    y_proba_poly = svc_poly.predict_proba(X_test)
    y_proba_rbf = svc_rbf.predict_proba(X_test)
    # # # # #
    # # # # # ------Evaluate the SVC models------
    evaluate_model(svc_linear, y_test, y_pred_svc_linear, y_proba_linear, model_name='SVC_Linear')
    plot_roc_ovr(y_train, y_test, y_proba_linear, model_name='SVC_Linear')
    plot_roc_ovo(y_train, y_test, y_proba_linear, model_name='SVC_Linear')
    stratified_kfold_cv(X, y, svc_linear, 'SVC_Linear')
    # # #
    evaluate_model(svc_poly, y_test, y_pred_svc_poly, y_proba_poly, model_name='SVC_Poly')
    plot_roc_ovr(y_train, y_test, y_proba_poly, model_name='SVC_Poly')
    plot_roc_ovo(y_train, y_test, y_proba_poly, model_name='SVC_Poly')
    stratified_kfold_cv(X, y, svc_poly, 'SVC_Poly')
    # # #
    evaluate_model(svc_rbf, y_test, y_pred_svc_rbf, y_proba_rbf, model_name='SVC_RBF')
    plot_roc_ovr(y_train, y_test, y_proba_rbf, model_name='SVC_RBF')
    plot_roc_ovo(y_train, y_test, y_proba_rbf, model_name='SVC_RBF')
    stratified_kfold_cv(X, y, svc_rbf, 'SVC_RBF')
    # #
    # # # # ------------Grid Search CV for SVM----------------
    print('------------Grid Search CV for SVM (One v/s Rest)----------------')
    svc = SVC(random_state=5805, probability=True)
    # svc_params = {'C': [0.1, 0.2],
    #               'kernel': ['poly', 'rbf']}
    # grid_svm = GridSearchCV(estimator=svc, param_grid=svc_params, scoring='f1_macro', n_jobs=-1)
    # grid_svm.fit(X_train, y_train)
    print('After Grid Search best parameters are:')
    # best_params_svc = grid_svm.best_params_
    best_params_svc = {'C': 0.2, 'kernel': 'rbf'}
    print(best_params_svc)
    # # # best_params_svc = {'C': 0.2, 'kernel': 'rbf'}
    grid_svm = SVC(**best_params_svc, random_state=5805, probability=True)
    grid_svm.fit(X_train, y_train)
    y_best_svc_pred = grid_svm.predict(X_test)
    y_best_svc_proba = grid_svm.predict_proba(X_test)
    evaluate_model(grid_svm, y_test, y_best_svc_pred, y_best_svc_proba, model_name='Grid Search SVM')
    plot_roc_ovr(y_train, y_test, y_best_svc_proba, model_name='Grid Search SVM')
    plot_roc_ovo(y_train, y_test, y_best_svc_proba, model_name='Grid Search SVM')
    stratified_kfold_cv(X, y, grid_svm, 'Grid Search SVM ')
    # #
    # # # # -------------------------------------------------------------------------
    # # # #           Classifier 5: Naive Bayes Classifier
    # # # # -------------------------------------------------------------------------
    gnb = GaussianNB()
    # # # Fit the model
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    y_pred_gnb_proba = gnb.predict_proba(X_test)
    # # #
    # # # # Evaluate the models
    evaluate_model(gnb, y_test, y_pred_gnb, y_pred_gnb_proba, 'GaussianNB')
    plot_roc_ovr(y_train, y_test, y_pred_gnb_proba, model_name='GaussianNB')
    plot_roc_ovo(y_train, y_test, y_pred_gnb_proba, model_name='GaussianNB')
    stratified_kfold_cv(X, y, gnb, 'GaussianNB')
    params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
    gs_NB = GridSearchCV(estimator=gnb,
                         param_grid=params_NB,
                         scoring='f1_macro')
    gs_NB.fit(X_train, y_train)
    best_params_NB = gs_NB.best_params_
    print('Best Parameters of Grid Search NB Classifier are:',best_params_NB)
    gs_NB = GaussianNB(**best_params_NB)
    gs_NB.fit(X_train, y_train)
    y_pred_gnb = gs_NB.predict(X_test)
    y_pred_gnb_proba = gs_NB.predict_proba(X_test)
    evaluate_model(gs_NB, y_test, y_pred_gnb, y_pred_gnb_proba, 'Grid Search GaussianNB')
    plot_roc_ovr(y_train, y_test, y_pred_gnb_proba, model_name='Grid Search GaussianNB')
    plot_roc_ovo(y_train, y_test, y_pred_gnb_proba, model_name='Grid Search GaussianNB')
    stratified_kfold_cv(X, y, gs_NB, 'Grid Search GaussianNB')
    #
    # # #
    # # # # -------------------------------------------------------------------------
    # # # #           Classifier 6: Random Forest Classifier
    # # # # # -------------------------------------------------------------------------
    rfc_model = RandomForestClassifier(n_estimators=50, random_state=5805)
    # # # Fit the model
    rfc_model.fit(X_train, y_train)
    # # # Predict credit_score
    y_pred_rfc = rfc_model.predict(X_test)
    y_pred_rfc_proba = rfc_model.predict_proba(X_test)
    # # #
    # # # # Evaluate Models
    evaluate_model(rfc_model, y_test, y_pred_rfc, y_pred_rfc_proba, model_name='Random Forest')
    plot_roc_ovr(y_train, y_test, y_pred_rfc_proba, model_name='Random Forest')
    plot_roc_ovo(y_train, y_test, y_pred_rfc_proba, model_name='Random Forest')
    stratified_kfold_cv(X, y, rfc_model, 'Random Forest Classifier')
    # # #
    # # # Grid Search for Random Forest Classifier
    print('------------------------Grid Search Random Forest Classifier-------------------------')
    # parameter_grid = {'n_estimators': [100, 125],
    #                   'criterion': ['gini', 'entropy']}
    # grid_rfc = GridSearchCV(estimator=rfc_model, param_grid=parameter_grid, scoring='f1_macro',n_jobs=-1)
    # grid_rfc.fit(X_train, y_train)
    # best_params = grid_rfc.best_params_
    best_params = {'criterion': 'gini', 'n_estimators': 125}
    print('R-Forest Classifier best parameters:', best_params)
    grid_rfc = RandomForestClassifier(**best_params, random_state=5805)
    grid_rfc.fit(X_train, y_train)
    # # # # Predict Credit Score with grid_rfc
    y_pred_rfc = grid_rfc.predict(X_test)
    y_pred_rfc_proba = grid_rfc.predict_proba(X_test)
    # # # # Evaluate Model
    evaluate_model(grid_rfc, y_test, y_pred_rfc, y_pred_rfc_proba, model_name='Grid Search RForest')
    plot_roc_ovr(y_train, y_test, y_pred_rfc_proba, model_name='Grid Search RForest')
    plot_roc_ovo(y_train, y_test, y_pred_rfc_proba, model_name='Grid Search RForest')
    stratified_kfold_cv(X, y, grid_rfc, 'Grid Search RForest')
    #
    # # ----------------------------------------------------------
    # #              Stacking Random Forest Classifier
    # # ----------------------------------------------------------
    print('--------------------Stacking with Random Forest Classifier--------------------')
    # # # # We will use a random forest classifier with logistic regression as final estimator
    base_estimators = [('rf', RandomForestClassifier(n_estimators=100, random_state=5805)),
                       ('rf2', RandomForestClassifier(n_estimators=100, random_state=5805))]
    stacked_model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(multi_class='ovr', random_state=5805),
                                       n_jobs=-1)
    # # #
    stacked_model.fit(X_train, y_train)
    y_pred_stacked = stacked_model.predict(X_test)
    y_pred_stacked_proba = stacked_model.predict_proba(X_test)
    # # # # Evaluate Model
    evaluate_model(stacked_model, y_test, y_pred_stacked, y_pred_stacked_proba, model_name='Stacked Random Forest')
    plot_roc_ovr(y_train, y_test, y_pred_stacked_proba, model_name='Stacked Random Forest')
    plot_roc_ovo(y_train, y_test, y_pred_stacked_proba, model_name='Stacked Random Forest')
    stratified_kfold_cv(X, y, stacked_model, 'Stacked Random Forest')
    #
    # # ----------------------------------------------------------
    # #              Bagging Random Forest Classifier
    # # ----------------------------------------------------------
    print('--------------------Bagging with Random Forest Classifier--------------------')
    rfc = RandomForestClassifier(n_estimators=100, random_state=5805)
    bag_classifier = BaggingClassifier(estimator=rfc, n_estimators=10, random_state=5805, n_jobs=-1)
    # # # # Fit the model
    bag_classifier.fit(X_train, y_train)
    # # #
    # # # # Predict the Credit_Score
    y_pred_bag = bag_classifier.predict(X_test)
    y_pred_bag_proba = bag_classifier.predict_proba(X_test)
    # # #
    # # # # Evaluate the model
    evaluate_model(bag_classifier, y_test, y_pred_bag, y_pred_bag_proba, model_name='Bagging Random Forest')
    plot_roc_ovr(y_train, y_test, y_pred_bag_proba, model_name='Bagging Random Forest')
    plot_roc_ovo(y_train, y_test, y_pred_bag_proba, model_name='Bagging Random Forest')
    stratified_kfold_cv(X, y, bag_classifier, 'Bagging Random Forest')
    #
    # # ----------------------------------------------------------
    # #              Boosting Random Forest Classifier
    # # ----------------------------------------------------------
    print('--------------------Boosting with Random Forest Classifier--------------------')
    rfc = RandomForestClassifier(n_estimators=125, random_state=5805)
    boost_classifier = AdaBoostClassifier(rfc, n_estimators=50, random_state=5805)
    boost_classifier.fit(X_train, y_train)
    # # # Predict the Credit_Score
    y_pred_boost = boost_classifier.predict(X_test)
    y_pred_boost_proba = boost_classifier.predict_proba(X_test)
    # # #
    # # # # Evaluate the model
    evaluate_model(boost_classifier, y_test, y_pred_boost, y_pred_boost_proba, model_name='Boosting Random Forest')
    plot_roc_ovr(y_train, y_test, y_pred_boost_proba, model_name='Boosting Random Forest')
    plot_roc_ovo(y_train, y_test, y_pred_boost_proba, model_name='Boosting Random Forest')
    stratified_kfold_cv(X, y, boost_classifier, 'Boosting Random Forest')

    # -----------------------------------------------------------
    #               Neural Network - MLP Classifier
    # -----------------------------------------------------------
    # 2 layers of 20 nodes each
    mlp_clf = MLPClassifier(hidden_layer_sizes=(20,)*2, activation='relu', max_iter=1000, random_state=5805)
    # # Fit the model
    mlp_clf.fit(X_train, y_train)
    # # Predict the Credit Score
    y_pred_mlp = mlp_clf.predict(X_test)
    y_pred_mlp_proba = mlp_clf.predict_proba(X_test)
    # # Evaluate the models
    evaluate_model(mlp_clf, y_test, y_pred_mlp, y_pred_mlp_proba,model_name='MLP Classifier')
    plot_roc_ovr(y_train, y_test, y_pred_mlp_proba, model_name='MLP Classifier')
    plot_roc_ovo(y_train, y_test,y_pred_mlp_proba, model_name='MLP Classifier')
    stratified_kfold_cv(X, y, mlp_clf, 'MLP Classifier')

    # Grid Search
    print('------------------------Grid Search MLP Classifier-------------------------')
    # parameters = {'hidden_layer_sizes': [(50,)*2, (50,)*1],
    #               'activation': ['relu', 'tanh'],
    #               'max_iter': [1000, 1200]}
    # grid_mlp = GridSearchCV(param_grid=parameters, estimator=MLPClassifier(random_state=5805), scoring='f1_macro',
    #                         n_jobs=-1)
    # grid_mlp.fit(X_train, y_train)
    # best_params = grid_mlp.best_params_
    best_params = {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'max_iter': 1000}
    print('Best Parameters for MLP Classifier:', best_params)
    # # # for MLP Classifier: {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'max_iter': 500, 'solver': 'adam'}
    grid_mlp = MLPClassifier(**best_params, random_state=5805)
    grid_mlp.fit(X_train, y_train)
    y_pred_mlp = grid_mlp.predict(X_test)
    y_pred_mlp_proba = grid_mlp.predict_proba(X_test)
    # # Evaluate Grid Search MLP
    evaluate_model(grid_mlp, y_test, y_pred_mlp, y_pred_mlp_proba, model_name='Grid Search MLP')
    plot_roc_ovr(y_train,y_test, y_pred_mlp_proba, model_name='Grid Search MLP')
    plot_roc_ovo(y_train, y_test, y_pred_mlp_proba, model_name='Grid Search MLP')
    stratified_kfold_cv(X, y, grid_mlp, 'Grid Search MLP')
    # ------------------------------------------------------------------------------------------------
    #                           Need to add K-fold cross validation (Code Ready!!!!)
    # --------------------------------------------------------------------------------------------

    # Print Table
    print(result_table)
    end=time.time()
    time_spent = end-start
    print('Execution time:',round((time_spent/60),2))

































