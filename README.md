## Model Training Results

<table>
    <tr>
        <td>Model</td>
        <td>Hyperparameters</td>
        <td>F1 Score (Train)</td>
        <td>F1 Score (Test)</td>
        <td>ROC AUC Score</td>
        <td>Threshold</td>
        <td>Comments</td>
    </tr>
    <tr>
        <td>Logistic Regression</td>
        <td>solver=liblinear, random_state=29</td>
        <td>0.42</td>
        <td>0.44</td>
        <td>0.89</td>
        <td>0.5</td>
        <td>&quot;High ROC AUC but low F1: the model ranks well but has issues with the threshold</td>
    </tr>
    <tr>
        <td>Logistic Regression</td>
        <td>solver=liblinear, random_state=29</td>
        <td>0.42</td>
        <td>0.44</td>
        <td>0.89</td>
        <td>0.5</td>
        <td>&quot;High ROC AUC but low F1: the model ranks well but has issues with the threshold</td>
    </tr>
    <tr>
        <td>Logistic Regression (Adjusted Threshold)</td>
        <td>solver=liblinear, random_state=29, threshold=0.08497651405362451</td>
        <td>0.42</td>
        <td>0.48</td>
        <td>0.89</td>
        <td>0.08</td>
        <td>Threshold optimization improved F1 from 0.44 to 0.48 by balancing Precision and Recall: FN decreased and TP increased, but FP also rose. ROC AUC remained at 0.89, indicating an unchanged ability of the model to distinguish classes. Overall, I would not use this model.</td>
    </tr>
    <tr>
        <td>KNN (n_neighbors=3)</td>
        <td>n_neighbors=3</td>
        <td>0.65</td>
        <td>0.4</td>
        <td>0.76</td>
        <td>0.5</td>
        <td>&quot;After GridSearchCV</td>
    </tr>
    <tr>
        <td>KNN (Adjusted Threshold, n_neighbors=3)</td>
        <td>n_neighbors=3, threshold=0.33</td>
        <td>0.65</td>
        <td>0.45</td>
        <td>0.76</td>
        <td>0.33</td>
        <td>&quot;The optimal threshold for KNN is 0.33. F1 increased to 0.45</td>
    </tr>
    <tr>
        <td>Decision Tree</td>
        <td>random_state=29</td>
        <td>1.0</td>
        <td>0.4</td>
        <td>0.66</td>
        <td>0.5</td>
        <td>&quot;Firstly</td>
    </tr>
    <tr>
        <td>Decision Tree (Optimized)</td>
        <td>{&#39;max_depth&#39;: 10, &#39;min_samples_leaf&#39;: 4, &#39;min_samples_split&#39;: 10}</td>
        <td>0.57</td>
        <td>0.49</td>
        <td>0.87</td>
        <td>0.5</td>
        <td>Optimization improved F1 and ROC AUC. FP was reduced, TP increased, but FN remains high. Next steps: optimize max_depth, min_samples_split, or add regularization and pruning to decrease FN.</td>
    </tr>
    <tr>
        <td>Decision Tree (Further Optimized)</td>
        <td>{&#39;max_depth&#39;: 5, &#39;min_samples_leaf&#39;: 2, &#39;min_samples_split&#39;: 3}</td>
        <td>0.44</td>
        <td>0.43</td>
        <td>0.85</td>
        <td>0.5</td>
        <td>The model has become more sensitive to the negative class but worse at recognizing positive cases. FP has decreased, but FN has increased. I will try adding regularization or pruning the tree.</td>
    </tr>
    <tr>
        <td>Decision Tree (Optimized with Pruning)</td>
        <td>{&#39;max_depth&#39;: 10, &#39;min_samples_leaf&#39;: 4, &#39;min_samples_split&#39;: 10}, ccp_alpha=0.001182291195789234</td>
        <td>0.51</td>
        <td>0.51</td>
        <td>0.84</td>
        <td>0.5</td>
        <td>&quot;The model with ccp_alpha pruning improved the balance between Precision and Recall</td>
    </tr>
    <tr>
        <td>Decision Tree (Optimized with Pruning)</td>
        <td>{&#39;max_depth&#39;: 10, &#39;min_samples_leaf&#39;: 4, &#39;min_samples_split&#39;: 10}, ccp_alpha=0.001182291195789234</td>
        <td>0.51</td>
        <td>0.51</td>
        <td>0.84</td>
        <td>0.5</td>
        <td>&quot;The model with ccp_alpha pruning improved the balance between Precision and Recall</td>
    </tr>
    <tr>
        <td>Random Forest (Optimized)</td>
        <td>{&#39;ccp_alpha&#39;: 0.0, &#39;max_depth&#39;: 20, &#39;min_samples_leaf&#39;: 1, &#39;min_samples_split&#39;: 2, &#39;n_estimators&#39;: 100}</td>
        <td>0.92</td>
        <td>0.45</td>
        <td>0.9</td>
        <td>0.5</td>
        <td>The model distinguishes classes well but has high FN (694), which lowers Recall. Next steps: lower the threshold to increase Recall and reduce FN.</td>
    </tr>
    <tr>
        <td>Random Forest (Optimized)</td>
        <td>{&#39;ccp_alpha&#39;: 0.0, &#39;max_depth&#39;: 20, &#39;min_samples_leaf&#39;: 1, &#39;min_samples_split&#39;: 2, &#39;n_estimators&#39;: 100}</td>
        <td>0.92</td>
        <td>0.45</td>
        <td>0.9</td>
        <td>0.5</td>
        <td>The model distinguishes classes well but has high FN (694), which lowers Recall. Next steps: lower the threshold to increase Recall and reduce FN.</td>
    </tr>
    <tr>
        <td>Random Forest (Adjusted Threshold)</td>
        <td>{&#39;ccp_alpha&#39;: 0.0, &#39;max_depth&#39;: 20, &#39;min_samples_leaf&#39;: 1, &#39;min_samples_split&#39;: 2, &#39;n_estimators&#39;: 100}, threshold=0.10749947540045453</td>
        <td>0.92</td>
        <td>0.51</td>
        <td>0.9</td>
        <td>0.11</td>
        <td>Adjusting the threshold improved the model&#39;s performance by balancing Precision and Recall. FN significantly decreased, which is important for positive cases, but FP is still high. Conclusion: the model is effective and suitable for use, but efforts should continue to reduce FP.</td>
    </tr>
    <tr>
        <td>Gradient Boosting (Randomized Search)</td>
        <td>{&#39;n_estimators&#39;: 100, &#39;min_samples_split&#39;: 10, &#39;min_samples_leaf&#39;: 1, &#39;max_depth&#39;: 5, &#39;learning_rate&#39;: 0.1}</td>
        <td>0.57</td>
        <td>0.5</td>
        <td>0.91</td>
        <td>0.5</td>
        <td>The model tends to miss positive cases; it is advisable to optimize the threshold to improve the balance between Precision and Recall.</td>
    </tr>
    <tr>
        <td>Gradient Boosting (Randomized Search, Adjusted Threshold)</td>
        <td>{&#39;n_estimators&#39;: 100, &#39;min_samples_split&#39;: 10, &#39;min_samples_leaf&#39;: 1, &#39;max_depth&#39;: 5, &#39;learning_rate&#39;: 0.1}, threshold=0.07434085884412608</td>
        <td>0.57</td>
        <td>0.5</td>
        <td>0.91</td>
        <td>0.07</td>
        <td>This model is better balanced; however, it is worth trying the Gradient Boosting model with Hyperopt (Bayesian Optimization) for further improvements.</td>
    </tr>
    <tr>
        <td>Gradient Boosting (Randomized Search)</td>
        <td>{&#39;n_estimators&#39;: 100, &#39;min_samples_split&#39;: 10, &#39;min_samples_leaf&#39;: 1, &#39;max_depth&#39;: 5, &#39;learning_rate&#39;: 0.1}</td>
        <td>0.57</td>
        <td>0.5</td>
        <td>0.91</td>
        <td>0.5</td>
        <td>The model tends to miss positive cases; it is advisable to optimize the threshold to improve the balance between Precision and Recall.</td>
    </tr>
    <tr>
        <td>Gradient Boosting (Randomized Search, Adjusted Threshold)</td>
        <td>{&#39;n_estimators&#39;: 100, &#39;min_samples_split&#39;: 10, &#39;min_samples_leaf&#39;: 1, &#39;max_depth&#39;: 5, &#39;learning_rate&#39;: 0.1}, threshold=0.07434085884412608</td>
        <td>0.57</td>
        <td>0.5</td>
        <td>0.91</td>
        <td>0.07</td>
        <td>This model is better balanced; however, it is worth trying the Gradient Boosting model with Hyperopt (Bayesian Optimization) for further improvements.</td>
    </tr>
    <tr>
        <td>Gradient Boosting (Hyperopt)</td>
        <td>{&#39;learning_rate&#39;: 0.09276117355569657, &#39;max_depth&#39;: 1, &#39;min_samples_leaf&#39;: 1, &#39;min_samples_split&#39;: 2, &#39;n_estimators&#39;: 1}</td>
        <td>0.56</td>
        <td>0.5</td>
        <td>0.91</td>
        <td>0.5</td>
        <td>The model effectively separates classes, but FN and FP remain problematic. After Bayesian optimization, the model is stable, but it is advisable to adjust the threshold to reduce errors.</td>
    </tr>
    <tr>
        <td>Gradient Boosting (Hyperopt, Adjusted Threshold)</td>
        <td>{&#39;learning_rate&#39;: 0.09276117355569657, &#39;max_depth&#39;: 1, &#39;min_samples_leaf&#39;: 1, &#39;min_samples_split&#39;: 2, &#39;n_estimators&#39;: 1}, threshold=0.09690758822832464</td>
        <td>0.56</td>
        <td>0.52</td>
        <td>0.91</td>
        <td>0.1</td>
        <td>The model performs well with a high ROC AUC Score, but increased false positives indicate a need for retuning to balance Precision and Recall, potentially by reducing estimators or applying regularization.</td>
    </tr>
</table>

