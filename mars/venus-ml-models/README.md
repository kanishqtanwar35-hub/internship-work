# ML models for Venus

## What it was
At MaRS I built predictive models for Venus, one of their internal software platforms. The work was the classic tabular ML workflow, and it's where I really learned that the model is the easy part.

## What I did
1. **Pulled historical data** from the platform.
2. **Cleaned it.** Missing values, typos in categories, outliers that were clearly data entry mistakes.
3. **Feature engineering.** For example turning dates into month or season, categories into numbers, and building ratio features between columns. This gave bigger gains than any model change.
4. **Compared three models** on the same data with cross-validation:
   - **Decision Tree**: easy to explain, but one tree overfits easily
   - **Random Forest**: many trees on random samples of the data, voting together, much more stable
   - **XGBoost**: trees built one after another, each one fixing the errors of the previous ones, usually the most accurate on this kind of data
5. **Tuned** the best one with a grid search over depth, learning rate and number of trees.

I kept the Decision Tree in the comparison on purpose. It's the baseline that shows how much the ensembles are really adding, and it's the easiest one to explain to non-technical people.

## What's in this folder
The real data stays with MaRS, so `train_compare.py` generates a synthetic project-task dataset (predicting whether a task gets delayed) with a similar shape. The workflow is the same: missing values, feature engineering, comparison with 5-fold cross-validation, then tuning.

```bash
python train_compare.py
```

On this synthetic data, XGBoost comes out ahead of Random Forest, which beats the single Decision Tree. That matches what I usually saw on the real data.
