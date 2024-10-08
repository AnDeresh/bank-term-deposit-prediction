## Model Training Results

| Model                                           | Hyperparameters                                                                                            |   F1 Score (Train) |   F1 Score (Test) |   ROC AUC Score |   Threshold           | Comments                                                                                                                                                                                                                                                                                |
|:------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|-------------------:|------------------:|----------------:|----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression                                       | solver=liblinear, random_state=29                                                                                                                       |               0.42 |              0.44 |            0.89 |        0.5  | High ROC AUC but low F1: the model ranks well but has issues with the threshold, leading to low Recall for the "Yes" class. It is worth adjusting the threshold to reduce False Negatives (FN). I would not use this model.                                                             |
| Logistic Regression                                       | solver=liblinear, random_state=29                                                                                                                       |               0.42 |              0.44 |            0.89 |        0.5  | High ROC AUC but low F1: the model ranks well but has issues with the threshold, leading to low Recall for the "Yes" class. It is worth adjusting the threshold to reduce False Negatives (FN). I would not use this model.                                                             |
| Logistic Regression (Adjusted Threshold)                  | solver=liblinear, random_state=29, threshold=0.08497651405362451                                                                                        |               0.42 |              0.48 |            0.89 |        0.08 | Threshold optimization improved F1 from 0.44 to 0.48 by balancing Precision and Recall: FN decreased and TP increased, but FP also rose. ROC AUC remained at 0.89, indicating an unchanged ability of the model to distinguish classes. Overall, I would not use this model.            |
| KNN (n_neighbors=3)                                       | n_neighbors=3                                                                                                                                           |               0.65 |              0.4  |            0.76 |        0.5  | After GridSearchCV, the optimal n_neighbors is 3. F1 = 0.40, indicating an imbalance between Precision and Recall for the "Yes" class. The confusion matrix shows a high FN, reducing Recall. It may be worthwhile to adjust the threshold or try other models.                         |
| KNN (Adjusted Threshold, n_neighbors=3)                   | n_neighbors=3, threshold=0.33                                                                                                                           |               0.65 |              0.45 |            0.76 |        0.33 | The optimal threshold for KNN is 0.33. F1 increased to 0.45, and the balance between Precision and Recall improved. Lowering the threshold increased "Yes" predictions, but FP also rose. Next steps: test other models to reduce FP.                                                   |
| Decision Tree                                             | random_state=29                                                                                                                                         |               1    |              0.4  |            0.66 |        0.5  | Firstly, the model is overfitting. It has limited ability to distinguish between classes. There is an issue with the balance between Precision and Recall for the "Yes" class. I would not use this model, but I will try to tune the hyperparameters for it.                           |
| Decision Tree (Optimized)                                 | {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}                                                                                       |               0.57 |              0.49 |            0.87 |        0.5  | Optimization improved F1 and ROC AUC. FP was reduced, TP increased, but FN remains high. Next steps: optimize max_depth, min_samples_split, or add regularization and pruning to decrease FN.                                                                                           |
| Decision Tree (Further Optimized)                         | {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 3}                                                                                         |               0.44 |              0.43 |            0.85 |        0.5  | The model has become more sensitive to the negative class but worse at recognizing positive cases. FP has decreased, but FN has increased. I will try adding regularization or pruning the tree.                                                                                        |
| Decision Tree (Optimized with Pruning)                    | {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}, ccp_alpha=0.001182291195789234                                                       |               0.51 |              0.51 |            0.84 |        0.5  | The model with ccp_alpha pruning improved the balance between Precision and Recall, leading to better identification of "Yes" classes. Pruning reduced overfitting and enhanced generalization. I would choose this model for further work.                                             |
| Decision Tree (Optimized with Pruning)                    | {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}, ccp_alpha=0.001182291195789234                                                       |               0.51 |              0.51 |            0.84 |        0.5  | The model with ccp_alpha pruning improved the balance between Precision and Recall, leading to better identification of "Yes" classes. Pruning reduced overfitting and enhanced generalization. I would choose this model for further work.                                             |
| Random Forest (Optimized)                                 | {'ccp_alpha': 0.0, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}                                                 |               0.92 |              0.45 |            0.9  |        0.5  | The model distinguishes classes well but has high FN (694), which lowers Recall. Next steps: lower the threshold to increase Recall and reduce FN.                                                                                                                                      |
| Random Forest (Optimized)                                 | {'ccp_alpha': 0.0, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}                                                 |               0.92 |              0.45 |            0.9  |        0.5  | The model distinguishes classes well but has high FN (694), which lowers Recall. Next steps: lower the threshold to increase Recall and reduce FN.                                                                                                                                      |
| Random Forest (Adjusted Threshold)                        | {'ccp_alpha': 0.0, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}, threshold=0.10749947540045453                  |               0.92 |              0.51 |            0.9  |        0.11 | Adjusting the threshold improved the model's performance by balancing Precision and Recall. FN significantly decreased, which is important for positive cases, but FP is still high. Conclusion: the model is effective and suitable for use, but efforts should continue to reduce FP. |
| Gradient Boosting (Randomized Search)                     | {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 5, 'learning_rate': 0.1}                                             |               0.57 |              0.5  |            0.91 |        0.5  | The model tends to miss positive cases; it is advisable to optimize the threshold to improve the balance between Precision and Recall.                                                                                                                                                  |
| Gradient Boosting (Randomized Search, Adjusted Threshold) | {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 5, 'learning_rate': 0.1}, threshold=0.07434085884412608              |               0.57 |              0.5  |            0.91 |        0.07 | This model is better balanced; however, it is worth trying the Gradient Boosting model with Hyperopt (Bayesian Optimization) for further improvements.                                                                                                                                  |
| Gradient Boosting (Randomized Search)                     | {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 5, 'learning_rate': 0.1}                                             |               0.57 |              0.5  |            0.91 |        0.5  | The model tends to miss positive cases; it is advisable to optimize the threshold to improve the balance between Precision and Recall.                                                                                                                                                  |
| Gradient Boosting (Randomized Search, Adjusted Threshold) | {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 5, 'learning_rate': 0.1}, threshold=0.07434085884412608              |               0.57 |              0.5  |            0.91 |        0.07 | This model is better balanced; however, it is worth trying the Gradient Boosting model with Hyperopt (Bayesian Optimization) for further improvements.                                                                                                                                  |
| Gradient Boosting (Hyperopt)                              | {'learning_rate': 0.09276117355569657, 'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1}                                |               0.56 |              0.5  |            0.91 |        0.5  | The model effectively separates classes, but FN and FP remain problematic. After Bayesian optimization, the model is stable, but it is advisable to adjust the threshold to reduce errors.                                                                                              |
| Gradient Boosting (Hyperopt, Adjusted Threshold)          | {'learning_rate': 0.09276117355569657, 'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1}, threshold=0.09690758822832464 |               0.56 |              0.52 |            0.91 |        0.1  | The model performs well with a high ROC AUC Score, but increased false positives indicate a need for retuning to balance Precision and Recall, potentially by reducing estimators or applying regularization.                                                                           |
