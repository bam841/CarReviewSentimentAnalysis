# Car Review Sentiment Analysis — Project Plan & Agent Context

## Project Overview
- **Dataset:** `car_reviews_.csv` (530 rows, 17 columns; `review_text` max length: 57 words, mean: ~35.7 words)
- **Notebook:** `CarSentimentAnalysis.ipynb`
- **Core Objective:** Build and compare two distinct sentiment analysis approaches on car reviews:
  1. **Semantic Model:** Reads full sentence context to determine sentiment.
  2. **Weighted Rule-Based Model:** Assigns weights to sentiment keywords (especially negative words) and classifies based on total weight balance.

---

## Planning Decisions
*(Mark your choices with `[x]` and add any custom details under **Notes**.)*

### 1. Ground-Truth Labels for the Semantic Model
Since `car_reviews_.csv` only has `review_text` and `rating` (no `sentiment` column), which setup should we use?
- [ ] **Option A:** Use `rating` *only* as the answer key (`label`) to fine-tune the semantic model on `review_text` (the model never sees `rating` as an input feature).
- [ ] **Option B:** Skip fine-tuning and run an already pre-trained sentiment transformer directly on `review_text` (zero `rating` usage).
- [ ] **Option C:** Use a pre-trained sentiment model to auto-label `review_text` first, then fine-tune/evaluate the models on those labels.

**Notes:** 
> 

---

### 2. Sentiment Classes
Which classification scheme should both models output?
- [ ] **Binary (Positive vs. Negative)** — Drop neutral/3-star reviews (e.g., *"adequate for most needs"*).
- [ ] **Binary (Positive vs. Negative)** — Keep 3-star reviews and assign them to a specific class (specify in Notes).
- [ ] **3-Class (Positive, Neutral, Negative)** — Keep all reviews across three sentiment labels.

**Notes:** 
> 

---

### 3. Transformer Architecture(s)
Your notebook imports both **RoBERTa** and **XLNet**. Which models should we build and compare?
- [ ] **Rule-Based vs. RoBERTa** (2 models)
- [ ] **Rule-Based vs. RoBERTa vs. XLNet** (3 models)
- [ ] **Other** (specify in Notes)

**Notes:** 
> 

---

### 4. Rule-Based Weighting Design
How should the weighted rule-based lexicon be constructed and evaluated?
- [ ] **Manual weights** curated from EDA word clouds & top word frequencies.
- [ ] **Automated weights** computed statistically from word frequencies / TF-IDF on the training split.
- [ ] **Hybrid** (automated baseline weights + manual fine-tuning).
- [ ] **Include negation handling** (e.g., flipping *"not terrible"* or *"no issues"*).
- [ ] **No negation handling** (strict word-weight summation only).

**Notes:** 
> 

---

### 5. Final Outputs & Comparisons
Beyond Accuracy and F1-score, which deliverables should be included at the end of the notebook?
- [ ] Classification reports & side-by-side metrics summary table
- [ ] Confusion matrices (visualized side-by-side)
- [ ] Error analysis table (reviews where Rule-Based fails vs. Semantic succeeds, and vice versa)
- [ ] Interactive inference cell to test new custom car reviews across all models
- [ ] Saved model artifacts / exported predictions CSV

**Notes:** 
> 

---

### 6. Hardware & Environment (`tf_gpu`)
What compute device is available for PyTorch in your current environment (`torch.cuda.is_available()`)?
- [ ] **GPU (`True`)** — CUDA is active for PyTorch.
- [ ] **CPU (`False`)** — PyTorch will run on CPU (will tune batch size, `max_length`, and epochs accordingly).
- [ ] **Not checked yet** — Include a hardware check cell first.

**Notes:** 
> 

---

## Additional Instructions / Constraints
*(Add any extra goals, rubric requirements, or coding style preferences here.)*
- 