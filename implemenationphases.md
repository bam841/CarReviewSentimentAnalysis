Phase 0 — Environment Verification, Reproducibility & Notebook Cleanup
Objective

Fix existing notebook syntax errors, lock deterministic seeds, and verify GPU and tokenizer dependencies before processing data.
Tasks

    Fix the IndentationError in the EDA Word Cloud cell and confirm it runs cleanly.

    Create a reproducibility utility set_seed(42) locking random, np.random, torch.manual_seed, torch.cuda.manual_seed_all, and torch.backends.cudnn.deterministic = True.

    Define DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") and print GPU name, VRAM, PyTorch version, and Transformers version.

    Verify sentencepiece is installed (required by XLNetTokenizer); install it if missing.

DO NOT

    Do NOT split data, define models, or download weights yet.

Verification Gate (Must Pass Before Stopping)

    assert torch.cuda.is_available(), "CUDA GPU not detected in tf_gpu"

    import sentencepiece succeeds without ImportError.

    Word Cloud cell executes with zero errors.

Phase 1 — Label Engineering, Feature Isolation & Stratified Split
Objective

Create the binary sentiment answer key from rating, isolate review_text as the sole model input, and lock in a reproducible 3-way split shared by all models.
Tasks

    Print full df["rating"].value_counts().sort_index() for audit.

    Filter out neutral reviews (df_model = df[df["rating"] != 3].copy()).

    Create binary target: df_model["label"] = (df_model["rating"] >= 4).astype(int) (0 = Negative, 1 = Positive).

    Strip leading/trailing whitespace on review_text ONLY (keep original casing, punctuation, and stopwords intact for RoBERTa/XLNet).

    Drop all columns except ["review_text", "label"].

    Perform a Stratified 70% Train / 15% Val / 15% Test split (random_state=42) into train_df, val_df, and test_df (reset indices on all three).

    Compute inverse-frequency class_weights strictly from train_df["label"] and initialize the global criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05).

DO NOT

    Do NOT strip punctuation, lowercase, or remove stopwords from review_text.

    Do NOT include rating or any metadata column inside train_df, val_df, or test_df.

    Do NOT build lexicons or tokenizers yet.

Verification Gate (Must Pass Before Stopping)

    assert list(train_df.columns) == ["review_text", "label"]

    assert set(train_df["label"].unique()) == {0, 1}

    assert len(set(train_df.index) & set(test_df.index)) == 0 (after verifying zero overlapping texts if unique).

    assert class_weights.shape == (2,)

    Print split row counts and exact class distribution (%) for Train, Val, and Test.

Phase 2 — Hybrid Rule-Based Lexicon Construction (Train-Split Only)
Objective

Build a transparent, leak-free weighted sentiment dictionary strictly from train_df that scores isolated words without negation handling (preserving the intended contrast against Semantic RoBERTa).
Tasks

    Tokenize words using lowercase alphabetic regex re.findall(r"[a-z]+", text.lower()) inside a helper function only (never overwrite train_df["review_text"]).

    Compute automated baseline word sentiment scores on train_df only (e.g., smoothed log-odds ratio or class-contrastive TF-IDF difference between Positive and Negative training reviews).

    Print the top 30 positive-scored and top 30 negative-scored words so non-sentiment noise (car brands/models such as porsche, genesis, tacoma, volkswagen, toyota) can be manually pruned into an explicit EXCLUDE_TERMS set and key sentiment weights refined (Hybrid design).

    Save the final dictionary HYBRID_LEXICON: dict[str, float] (positive words > 0.0, negative words < 0.0).

    Implement extract_rule_features(text, lexicon):

        Sums all positive word weights -> pos_weight_sum (≥0≥0).

        Sums all negative word weights -> neg_weight_sum (≤0≤0).

        Computes net_score = pos_weight_sum + neg_weight_sum.

        Returns (pos_weight_sum, neg_weight_sum, net_score, matched_word_dict).

        NO negation handling: Do NOT flip signs for "not", "no", "never", etc.

DO NOT

    Do NOT fit TF-IDF or word frequencies on df, val_df, or test_df.

    Do NOT add negation-flipping rules.

    Do NOT build any PyTorch neural network modules yet.

Verification Gate (Must Pass Before Stopping)

    Print HYBRID_LEXICON size (number of positive vs. negative words).

    Run sanity check on "The car is not bad":

        assert extract_rule_features("The car is not bad", HYBRID_LEXICON)[2] < 0, confirming isolated word weighting leans Negative when "bad" appears.

    Run sanity check on a review with 2 negative words and 1 positive word: confirm net_score < 0.

Phase 3 — Shared Native PyTorch Training & Evaluation Engine
Objective

Write one clean, reusable native PyTorch training/evaluation engine implementing the explicit forward pass, Cross-Entropy loss, backpropagation, gradient clipping, and AdamW update cycle.
Tasks

    Implement train_one_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device, use_amp=True):

        Sets model.train().

        Loop per batch:

            optimizer.zero_grad(set_to_none=True)

            Forward pass inside torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) -> raw logits of shape (B, 2).

            Loss: loss = criterion(logits, labels).

            Backward pass: scaler.scale(loss).backward().

            Unscale + Clip: scaler.unscale_(optimizer) -> torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).

            Update: scaler.step(optimizer) -> scaler.update() -> scheduler.step().

        Returns (avg_train_loss, train_macro_f1, train_acc).

    Implement evaluate(model, dataloader, criterion, device):

        Sets model.eval() and wraps in with torch.no_grad():.

        Forward pass -> raw logits -> compute val_loss = criterion(logits, labels).

        Compute probs = torch.softmax(logits, dim=-1) and preds = probs.argmax(dim=-1).

        Returns dict: {"loss", "acc", "macro_f1", "probs", "preds", "labels"}.

    Implement run_two_stage_training(...) helper supporting:

        Stage 1 (Head Warmup): Freeze backbone, train classification head (epochs=2, lr=1e-3).

        Stage 2 (End-to-End Fine-Tuning): Unfreeze backbone, train (max_epochs=6, lr=2e-5, EarlyStopping patience=2 on val_loss), save best checkpoint (torch.save(model.state_dict(), save_path)), and reload best weights before returning history.

DO NOT

    Do NOT instantiate XLNet or RoBERTa yet.

    Do NOT apply Softmax inside any loss computation.

Verification Gate (Must Pass Before Stopping)

    Run a 2-batch dummy tensor unit test (nn.Linear(8, 2)) through train_one_epoch and evaluate:

        Assert loss is finite (not torch.isnan(loss)).

        Assert returned probs.shape == (N, 2) and torch.allclose(probs.sum(dim=-1), torch.ones(N), atol=1e-5).

Phase 4 — Model 1: XLNet Baseline (xlnet-base-cased)
Objective

Fine-tune XLNet as the baseline semantic benchmark using 2-stage fine-tuning and cache its validation/test Softmax probabilities.
Activation Function Contract (Model 1)

    Encoder Hidden Layers: Pre-trained GELU (config.ff_activation == "gelu"). DO NOT override.

    Sequence Summary Layer: Pre-trained Tanh (config.summary_activation == "tanh").

    Output Head: Linear projection to 2 raw logits -> Softmax only in evaluate().

Tasks

    Load XLNetTokenizer.from_pretrained("xlnet-base-cased") and build TransformerReviewDataset (max_length=128, dynamic padding via DataCollatorWithPadding, batch_size=16).

    Instantiate XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2) and move to DEVICE.

    Note on AMP stability: Use bfloat16 if torch.cuda.is_bf16_supported(), else set use_amp=False for XLNet if fp16 produces NaN gradients.

    Run 2-stage fine-tuning (freeze model.transformer in Stage 1 for 2 epochs at lr=1e-3; unfreeze all in Stage 2 for up to 6 epochs at lr=2e-5, patience=2).

    Save best checkpoint to checkpoints/best_xlnet.pt.

    Reload best checkpoint and store val_res_xlnet and test_res_xlnet (including probs arrays of shape (N, 2)).

DO NOT

    Do NOT modify model.config.ff_activation or add Softmax to forward().

    Do NOT start RoBERTa yet.

Verification Gate (Must Pass Before Stopping)

    assert model_xlnet.config.ff_activation == "gelu"

    assert val_res_xlnet["probs"].shape == (len(val_df), 2)

    assert test_res_xlnet["probs"].shape == (len(test_df), 2)

    assert np.allclose(test_res_xlnet["probs"].sum(axis=1), 1.0, atol=1e-5)

    Print XLNet Train/Val Loss history and Baseline test_df Accuracy + Macro F1.

Phase 5 — Model 2: Rule-Based RoBERTa (RuleBasedRobertaClassifier)
Objective

Build a custom PyTorch nn.Module powered by roberta-base static word embeddings + HYBRID_LEXICON word weights without Transformer self-attention, so classification strictly reflects weighted word balance rather than sentence syntax.
Activation Function & Architecture Contract (Model 2)

    Embedding Backbone: Extract ONLY roberta.embeddings.word_embeddings (shape vocab_size x 768) from RobertaModel.from_pretrained("roberta-base").

        CRITICAL: Freeze word_embeddings (requires_grad = False) and DO NOT call roberta.encoder (zero self-attention layers).

    Lexicon-Weighted Token Pooling:

        Map each review's tokens to their signed lexicon weights from HYBRID_LEXICON.

        Compute weighted embedding pool over matched tokens (falling back to mean token embedding over non-padding tokens if zero lexicon words match) -> vector emb_pool of shape (B, 768).

    Scalar Rule Vector Normalization — Tanh:

        Compute raw rule tensor rule_raw = [pos_weight_sum, neg_weight_sum, net_score] of shape (B, 3).

        Apply temperature-scaled Tanh (rule_norm = torch.tanh(rule_raw / 3.0)) to bound rule magnitudes cleanly into [-1, +1] without early saturation while preserving positive/negative signs.

    Custom Classification MLP Head — ReLU:
    Python

    self.classifier = nn.Sequential(
        nn.Linear(768 + 3, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2)
    )

    Forward Output: Returns raw logits (B, 2) -> Softmax only in evaluate().

Tasks

    Build RuleBasedRobertaDataset returning input_ids, attention_mask, token_lex_weights (float tensor aligned to input_ids), rule_raw (float tensor of shape (3,)), and label.

    Implement RuleBasedRobertaClassifier(nn.Module) following the exact contract above.

    Train using train_one_epoch + evaluate with torch.optim.AdamW(lr=1e-3, weight_decay=0.01) for up to 15 epochs with Early Stopping (patience=3 on val_loss).

    Save best checkpoint to checkpoints/best_roberta_rule.pt.

    Reload best checkpoint and store val_res_rule and test_res_rule.

DO NOT

    Do NOT pass tokens through RoBERTa's 12 Transformer attention layers.

    Do NOT add negation handling to token_lex_weights or rule_raw.

    Do NOT apply Softmax inside forward().

Verification Gate (Must Pass Before Stopping)

    assert not hasattr(model_rule, "encoder"), confirming zero self-attention blocks.

    Assert word_embeddings.weight.requires_grad == False.

    Run inference on "The car is not bad": print rule_raw, rule_norm, and Softmax [P(Neg), P(Pos)] — verify it leans Negative due to the isolated negative word "bad".

    Assert test_res_rule["probs"].shape == (len(test_df), 2) and print test_df Accuracy + Macro F1.

Phase 6 — Model 3: Semantic RoBERTa (roberta-base)
Objective

Fine-tune the full 12-layer RobertaForSequenceClassification model so multi-head self-attention reads full phrase structure and resolves negation/context (e.g., "the car is not bad" -> Positive).
Activation Function Contract (Model 3)

    Encoder Hidden Layers: Pre-trained GELU (config.hidden_act == "gelu"). DO NOT override.

    Multi-Head Self-Attention: Native Softmax attention across full sequence.

    Classification Head (RobertaClassificationHead): Native Dropout -> Linear(768, 768) -> torch.tanh() -> Dropout -> Linear(768, 2).

    Forward Output: Raw logits (B, 2) -> Softmax only in evaluate().

Tasks

    Build RobertaSemanticDataset (max_length=128, batch_size=16, dynamic padding).

    Instantiate RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2) and move to DEVICE.

    Run 2-stage fine-tuning:

        Stage 1 (Head Warmup): Freeze model.roberta, train model.classifier for 2 epochs at lr=1e-3.

        Stage 2 (Full Fine-Tuning): Unfreeze all parameters, train up to 6 epochs at lr=2e-5 with AdamW, AMP, and Early Stopping (patience=2 on val_loss).

    Save best checkpoint to checkpoints/best_roberta_semantic.pt.

    Reload best checkpoint and store val_res_sem and test_res_sem.

    Plot side-by-side Train vs. Validation Loss and Validation Macro F1 curves for Models 1, 2, and 3 to audit overfitting.

DO NOT

    Do NOT override config.hidden_act or add Softmax inside forward().

    Do NOT compute ensemble weights yet.

Verification Gate (Must Pass Before Stopping)

    assert model_sem.config.hidden_act == "gelu"

    assert test_res_sem["probs"].shape == (len(test_df), 2)

    Run inference on "The car is not bad": print Softmax [P(Neg), P(Pos)] and compare directly against Phase 5's Rule-Based RoBERTa output.

    Print test_df Accuracy + Macro F1 and display overfitting diagnostic plots.

Phase 7 — Weighted Soft-Voting Ensemble, Evaluation Suite & Interactive Tester
Objective

Combine calibrated Softmax probability distributions from Models 1, 2, and 3 via validation-optimized soft voting, then generate the full comparative benchmark and error analysis.
Activation & Combination Contract (Final Ensemble)

    Inputs: Post-Softmax probability matrices P1 (XLNet), P2 (Rule-Based RoBERTa), P3 (Semantic RoBERTa), each of shape (N, 2).

    Weighted Combination:
    Python

    P_ens = w1 * P1 + w2 * P2 + w3 * P3   # constraint: w1 + w2 + w3 == 1.0, wi >= 0
    preds = P_ens.argmax(axis=-1)

    FORBIDDEN: Do NOT apply a second Softmax to P_ens.

Tasks

    Weight Optimization (Validation-Only): Grid-search (w1, w2, w3) in increments of 0.05 (with w1 + w2 + w3 == 1.0) strictly on val_df to maximize Macro F1 (tie-break by lowest log-loss).

    Test Evaluation: Apply the frozen (w1, w2, w3) to test_df probabilities to produce test_res_ens.

    Final Deliverables:

        Metrics Comparison Table: Accuracy, Macro F1, Precision, Recall, and Test Loss across all 4 systems (XLNet Baseline, Rule-Based RoBERTa, Semantic RoBERTa, Final Ensemble).

        2x2 Confusion Matrix Grid: Visualized side-by-side on test_df.

        Error Analysis Table: Display exact test_df reviews where Rule-Based RoBERTa failed while Semantic RoBERTa / Ensemble succeeded (plus any cases where all models failed).

        Interactive Inference Function (predict_all_models(review_text)): Accepts any custom string (pre-loaded with "The car is not bad" and a mixed-sentiment car review) and prints:

            Rule-Based matched words + individual weights + net_score

            Softmax [P(Neg), P(Pos)] and predicted class for all 4 models.

        Artifact Export: Save predictions_comparison.csv containing review_text, true_label, and predictions/probabilities from all 4 models.

Verification Gate (Final Sign-Off)

    assert np.isclose(w1 + w2 + w3, 1.0)

    assert np.allclose(P_ens_test.sum(axis=1), 1.0, atol=1e-5)

    Confirm all 4 models were evaluated on the exact same len(test_df) samples (assert np.array_equal(...) on labels).


---

### How to Drive Your Coding Agent Without Hallucinations
Save the block above as `agents.md` in your repo root, and feed your agent **one short prompt per turn**:
* *"Read `agents.md` and execute **ONLY Phase 0**. Stop at the Verification Gate and show me the outputs."*
* Once Phase 0 passes: *"Phase 0 verified. Execute **ONLY Phase 1** from `agents.md` and stop at the Verification Gate."*

Whenever you want to execute **Phase 0** right here or add your professor's extra requirement into the spec first, just let me know.