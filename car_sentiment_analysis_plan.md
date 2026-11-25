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
- [x] **Option A:** Use `rating` *only* as the answer key (`label`) to fine-tune the semantic model on `review_text` (the model never sees `rating` as an input feature).
    I dont want the model to rely on the rating (1,2,3,4,5,) what i want to happen is for the model to be able to classify which is a good rating or a bad rating, since we are going to create three, and make an ensemble, one is the xlnet. it will serve as our base model where it will be the baseline for the performance of the model that we are going to create, then after that we will have another model that uses Roberta model, what we will do is use this roberta model and make it read each word in the text review, it will have two modelmodel ome is sstill roberta but it is a rule based and then the other one is the semantic type thgat still uses the ROberta model


  # what to do for the two ROberta model
    ## rule based description
     the rule based model, it will weight each words in the sentence to and then evaluate whether the text review is on the negative side or vvice versa. how it should works(for example i have two negative words in the text review and one postive word. then the prediction of the model would then lean heavily on the negative side)


     ## Semantic based description

     the semantci ROberta based model will be the one that reads the whole phrase even if there are negative words it can still be positive based on how it is constructed this is an analogy
           
           -- semantic analog sample:
                  the car is not bad 
                    as you can see there is a negative word there which is(bad) so in the rule based it will see it as negative right since there is a negative word that exists in the text review. but since it is a semantic model it will read the whole phrase and not just rely if it sees a negative word since  the phrase says the car is not bad, it means that the car is good
              
            

- [ ] **Option B:** Skip fine-tuning and run an already pre-trained sentiment transformer directly on `review_text` (zero `rating` usage).
- [] **Option C:** Use a pre-trained sentiment model to auto-label `review_text` first, then fine-tune/evaluate the models on those labels.

**Notes:** 
> 

---

### 2. Sentiment Classes
Which classification scheme should both models output?
- [x ] **Binary (Positive vs. Negative)** — Drop neutral/3-star reviews (e.g., *"adequate for most needs"*).
  note!! again we are not going to base on the rating label so dont focus on that we are going to do what i instructred you in the descrioptopn above
- [ ] **Binary (Positive vs. Negative)** — Keep 3-star reviews and assign them to a specific class (specify in Notes).
- [ ] **3-Class (Positive, Neutral, Negative)** — Keep all reviews across three sentiment labels.

**Notes:** 


---

### 3. Transformer Architecture(s)
Your notebook imports both **RoBERTa** and **XLNet**. Which models should we build and compare?
- [ ] **Rule-Based vs. RoBERTa** (2 models)
- [x] **Rule-Based vs. RoBERTa vs. XLNet** (3 models)
- [ ] **Other** (specify in Notes)

**Notes:** 
>  if the rule based can also used with Roberta please do so 

---

### 4. Rule-Based Weighting Design
How should the weighted rule-based lexicon be constructed and evaluated?
- [ ] **Manual weights** curated from EDA word clouds & top word frequencies.
- [ ] **Automated weights** computed statistically from word frequencies / TF-IDF on the training split.
- [x] **Hybrid** (automated baseline weights + manual fine-tuning).
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
- [x] **GPU (`True`)** — CUDA is active for PyTorch.
- [ ] **CPU (`False`)** — PyTorch will run on CPU (will tune batch size, `max_length`, and epochs accordingly).
- [ ] **Not checked yet** — Include a hardware check cell first.

**Notes:** 
> 

---

## Additional Instructions / Constraints
*(Add any extra goals, rubric requirements, or coding style preferences here.)*
- 
# agents.md — Car Review Sentiment Analysis (Phase-Gated ML Engineering Spec)

## 0. Agent Operating Protocol (Anti-Hallucination Rules)
You are acting as a Senior Machine Learning Engineer working inside `CarSentimentAnalysis.ipynb` using `car_reviews_.csv` (530 rows, `tf_gpu` CUDA environment).

**STRICT EXECUTION RULES:**
1. **ONE PHASE AT A TIME:** Execute **ONLY** the single phase explicitly requested by the user. **NEVER** write code, helper functions, or placeholder cells for future phases.
2. **STOP AT THE GATE:** At the end of each phase, run all **Verification Gate** assertions and print checks, report the observed outputs, and **STOP** to wait for user approval before proceeding.
3. **NO SILENT SUBSTITUTIONS:** Do **NOT** replace the native PyTorch training loop with Hugging Face `Trainer`. Do **NOT** import deprecated `transforders.AdamW` (always use `torch.optim.AdamW`).
4. **ZERO DATA LEAKAGE:** `rating` is used **ONLY** in Phase 1 to generate binary ground-truth `label` (`0`/`1`), then immediately excluded from all model inputs. Lexicons, class weights, and ensemble weights must **NEVER** touch `test_df`.
5. **NO DOUBLE ACTIVATIONS:** Follow the Global Activation & Loss Contract below without exception.

---

## 1. Global PyTorch Activation, Tensor Shape & Loss Contract
*(Binding across Models 1, 2, and 3)*

- **Input Feature (`X`):** `review_text` ONLY (tokenized with `max_length=128`, since dataset max word count is 57).
- **Target Label (`y`):** Binary integer tensor `torch.long` of shape `(batch_size,)` where `0 = Negative` (`rating <= 2`) and `1 = Positive` (`rating >= 4`).
- **Model Forward Output:** Every model's `forward()` **MUST** return **raw unnormalized logits** of shape `(batch_size, 2)` representing `[logit_neg, logit_pos]`.
  - **FORBIDDEN:** Never call `torch.softmax()`, `F.softmax()`, `torch.sigmoid()`, or `nn.Softmax()` inside `model.forward()`.
- **Loss Function (Shared by ALL 3 Models):**
  ```python
  criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
  and please make sure that the learning rate is just at  tehe rigth amount that fits with the alogrithm that is going to be used so taht the model gets higher accuracy d