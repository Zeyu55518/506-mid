# 506-mid
# Music Review Rating Prediction

**Author:** Zeyu Dong 

**Kaggle Name** Zeyu Dong

**Final Grade** 0.50375(Private)

**Goal:** Predict music review scores (1–5) using multimodal features and classical machine learning models (no boosting allowed).

---

##  1. Data Exploration

The dataset contains **Music Reviews**, including:

- **Text fields:** `summary`, `reviewText`  
- **Categorical field:** `genres`  
- **Numerical fields:** `TotalVotes`, `VoteHelpful`  
- **Target:** `Score` (1–5 star rating)

After cleaning missing values, ~294,000 labeled samples remain.

I split the dataset into:
- **Training set (80%)**
- **Validation set (20%)**, stratified by `Score`
- **Test set:** rows with missing scores for submission

```python
train = pd.read_csv("train.csv")
labeled = train.dropna(subset=["Score"]).copy()
test = train[train["Score"].isna()].copy()

X_tr, X_va, y_tr, y_va = train_test_split(
    labeled, labeled["Score"].astype(int),
    test_size=0.2, random_state=42, stratify=labeled["Score"]
)
```
---

## 2. Feature Extraction / Engineering

This project integrates **three modalities** of features:  
1. Numerical (metadata-based)  
2. Genre (categorical)  
3. Text (linguistic and semantic)  

All are standardized or vectorized appropriately and later concatenated into a single sparse matrix.

---


###  (1) Numerical Features

Four handcrafted numerical features were extracted from the metadata columns:

| Feature | Description |
|----------|-------------|
| `helpfulness_ratio` | Ratio of `VotedHelpful / TotalVotes` |
| `helpfulness_log` | Log-transformed ratio for stability |
| `review_length` | Number of tokens in `reviewText` |
| `num_exclaim` | Count of exclamation marks (`!`) |

Each is computed and standardized (mean=False to preserve sparsity):

```python
for df in [labeled, test]:
    df["helpfulness_ratio"] = np.where(df["TotalVotes"] > 0,
                                       df["VotedHelpful"] / df["TotalVotes"], 0)
    df["helpfulness_log"] = np.log1p(df["helpfulness_ratio"])
    df["review_length"] = df["reviewText"].fillna("").apply(lambda x: len(x.split()))
    df["num_exclaim"] = df["reviewText"].fillna("").apply(lambda x: x.count("!"))

scaler = StandardScaler(with_mean=False)
num_cols = ["helpfulness_ratio", "helpfulness_log", "review_length", "num_exclaim"]
Xtr_num = scaler.fit_transform(labeled.loc[X_tr.index, num_cols])
Xva_num = scaler.transform(labeled.loc[X_va.index, num_cols])
Xte_num = scaler.transform(test[num_cols])
```

 **Output shape:** `(294066, 4)`

---

###  (2) Genre Features (Categorical Encoding)

Genres are often given as comma-separated strings, e.g., `"Rock, Pop"`.  
To capture this categorical information, the pipeline:

1. Tokenizes all unique genre strings into word tokens (e.g., `rock`, `pop`).  
2. Filters rare tokens (min frequency ≥ 5).  
3. Applies **TF-IDF (1–2 gram)** vectorization to the joined genre text.

```python
vec_genre = TfidfVectorizer(max_features=2000, ngram_range=(1, 2),
                            min_df=2, sublinear_tf=True)
Xtr_genre = vec_genre.fit_transform(labeled.loc[X_tr.index, "genres_text"])
Xva_genre = vec_genre.transform(labeled.loc[X_va.index, "genres_text"])
Xte_genre = vec_genre.transform(test["genres_text"])
```

 **Genre TF-IDF Shape:** `(294066, 2000)`

---

###  (3) Text Features (Core Linguistic Signals)

Each review text is formed by merging the `summary` and `reviewText` columns:

```python
labeled["text"] = (labeled["summary"].fillna("") + " " + labeled["reviewText"].fillna("")).str.lower()
test["text"] = (test["summary"].fillna("") + " " + test["reviewText"].fillna("")).str.lower()
```

We then create **multi-view TF-IDF representations**:

| View | Analyzer | n-gram | Max Features | Purpose |
|------|-----------|--------|---------------|----------|
| Word | `word` | (1,4) | 50,000 | Capture local context |
| Char | `char_wb` | (3,6) | 20,000 | Capture subword patterns |
| Stem | `word` | (1,2) | 20,000 | Capture lemma variations |

```python
stemmer = SnowballStemmer("english")
def stem_text(x): return " ".join(stemmer.stem(w) for w in x.split())

X_tr_stem = X_tr["text"].apply(stem_text)
X_va_stem = X_va["text"].apply(stem_text)
X_te_stem = X_te["text"].apply(stem_text)

vec_word = TfidfVectorizer(max_features=50000, ngram_range=(1,4),
                           min_df=2, max_df=0.99, sublinear_tf=True)
vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6),
                           max_features=20000, min_df=3, binary=True)
vec_stem = TfidfVectorizer(max_features=20000, ngram_range=(1,2),
                           min_df=2, sublinear_tf=True)

Xtr_tfidf = sparse.hstack([
    vec_word.fit_transform(X_tr["text"]),
    vec_char.fit_transform(X_tr["text"]),
    vec_stem.fit_transform(X_tr_stem)
])
```

 **Text TF-IDF shape:** `(294066, 90000)`

---

### ⚙️ Dimensionality Reduction & Fusion

Since the total text dimension (90k) is too large, two techniques are applied:

#### (a) Chi-Square Feature Selection
Selects 15,000 most informative word/char/stem features based on `chi2` relevance to target labels.
```python
sel = SelectKBest(chi2, k=15000)
Xtr_txt_sel = sel.fit_transform(Xtr_tfidf, y_tr)
Xva_txt_sel = sel.transform(Xva_tfidf)
Xte_txt_sel = sel.transform(Xte_tfidf)
```

#### (b) SVD Compression
Performs **Truncated SVD** to capture latent semantic structure (2000 components):
```python
svd = TruncatedSVD(n_components=2000, random_state=42)
Xtr_svd = svd.fit_transform(Xtr_tfidf)
Xva_svd = svd.transform(Xva_tfidf)
Xte_svd = svd.transform(Xte_tfidf)
```

---

###  Final Feature Fusion

All features are concatenated horizontally:

```python
Xtr_f = sparse.hstack([
    Xtr_txt_sel, Xtr_genre,
    sparse.csr_matrix(Xtr_num_scaled),
    sparse.csr_matrix(Xtr_svd)
], format="csr")

Xva_f = sparse.hstack([
    Xva_txt_sel, Xva_genre,
    sparse.csr_matrix(Xva_num_scaled),
    sparse.csr_matrix(Xva_svd)
], format="csr")

Xte_f = sparse.hstack([
    Xte_txt_sel, Xte_genre,
    sparse.csr_matrix(Xte_num_scaled),
    sparse.csr_matrix(Xte_svd)
], format="csr")
```

 **Final Feature Shape:** `(294066, 19004)`

---

##  3. Model Creation and Assumptions

Two base models and one meta-stacking layer were used:

###  (1) Logistic Regression (Base Model A)
```python
lr = LogisticRegression(
    solver="lbfgs", multi_class="multinomial",
    class_weight="balanced", C=0.2, max_iter=4000
)
lr.fit(Xtr_f, y_tr)
```
- Handles multimodal sparse data efficiently  
- Regularized for class balance (`balanced` weighting)  
- **Macro-F1:** 0.4928

---

###  (2) Complement Naive Bayes (Base Model B)
```python
cnb = ComplementNB()
cnb.fit(Xtr_tfidf, y_tr)
```
- Trained **only on text TF-IDF**
- Robust under class imbalance (variant of NB)
- Provides class probability estimates for stacking

---

###  (3) Meta Logistic Regression (Stacking Layer)
The meta model learns how to optimally combine base predictions:

```python
probs_tr = lr.predict_proba(Xtr_f)
probs_va = lr.predict_proba(Xva_f)
probs_cnb_tr = cnb.predict_proba(Xtr_tfidf)
probs_cnb_va = cnb.predict_proba(Xva_tfidf)

scaler = StandardScaler(with_mean=False)
probs_tr_scaled = scaler.fit_transform(probs_tr)
probs_va_scaled = scaler.transform(probs_va)

Xtr_stack = sparse.hstack([Xtr_f, probs_tr_scaled, probs_cnb_tr])
Xva_stack = sparse.hstack([Xva_f, probs_va_scaled, probs_cnb_va])

meta = LogisticRegression(C=0.2, solver="lbfgs",
                          multi_class="multinomial",
                          class_weight="balanced", max_iter=2000)
meta.fit(Xtr_stack, y_tr)
```

 **Stacked F1:** 0.4961  
The meta-learner slightly improves balance between overconfident and underrepresented classes.

---

##  4. Model Tuning

###  (a) Log-Bias Optimization

After training the stacked model, a **bias correction step** was added on top of the log probabilities:

```python
proba = meta.predict_proba(Xva_stack) + 1e-12
logp = np.log(proba)

def eval_bias(B):
    p = np.exp(logp + B)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1) + classes.min()
    f1 = f1_score(y_va, pred, average="macro")
    return f1, B
```

Bias grid explored across five class dimensions using 32-core parallelization:

```python
bias_grid = np.linspace(-0.3, 0.3, 13)
results = Parallel(n_jobs=32, backend="loky", verbose=10)(
    delayed(eval_bias)(B)
    for B in itertools.product(bias_grid, repeat=5)
)
best_f1, best_B = max(results, key=lambda x: x[0])
```

 **Best Macro-F1 = 0.5011**  
 **Best Bias:** `[-0.3, -0.25, 0.3, 0.3, 0.3]`

---

##  5. Inference & Submission

Final inference merges all probabilities, applies scaling and optimized bias:

```python
probs_te = lr.predict_proba(Xte_f)
probs_te_scaled = scaler.transform(probs_te)
probs_cnb_te = cnb.predict_proba(Xte_tfidf)
Xte_stack = sparse.hstack([Xte_f, probs_te_scaled, probs_cnb_te])

probs_te = meta.predict_proba(Xte_stack) + 1e-12
logp = np.log(probs_te)
p_adj = np.exp(logp + best_B)
p_adj = p_adj / p_adj.sum(axis=1, keepdims=True)
test_pred = p_adj.argmax(axis=1) + 1

submission = pd.DataFrame({"id": test["id"], "Score": test_pred})
submission.to_csv("submission_f.csv", index=False)
```

 **Output file:** `submission_f.csv`  
 **Final Macro-F1:** `0.5011`

---

##  6. Evaluation Summary

| Model | Input | Macro-F1 | Accuracy |
|--------|--------|-----------|-----------|
| Base LR | TF-IDF + dense + genre | 0.4928 | 0.589 |
| LR + CNB Stacking | Prob. Fusion | 0.4961 | 0.59 |
| + Bias Optimization | + Log-bias Calibration | **0.5011** | ~0.60 |

---



