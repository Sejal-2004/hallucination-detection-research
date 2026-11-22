# Can Small Language Models Tell When They Are Wrong?

Training-free error detection for small open language models answering science questions, with and without retrieved context. Final semester research project (IT).

- **Live dashboard:** https://hallucination-detection-dashboard.streamlit.app/
- **Dashboard code:** https://github.com/Sejal-2004/hallucination-detection-dashboard

## Aim

To compare two training-free error detection methods, **answer agreement (D0)** and **semantic consistency (D1)**, for two small open language models answering ARC-Challenge multiple-choice science questions, with and without retrieved context.

An incorrect answer here means a selected option that differs from the dataset answer key. This is an answer-error measure and does not claim that every wrong answer is a hallucination in the broader factual sense.

## Research questions

1. How accurately do the models answer ARC-Challenge questions, with and without retrieval?
2. How well do D0 and D1 separate incorrect answers from correct ones, and does this change with retrieval?
3. How often do the models repeat the same incorrect answer across samples, and what does this mean for consistency-based detection?

## Method

| Part | Details |
|---|---|
| Dataset | ARC-Challenge. 150 dev questions (validation split) and 300 test questions (test split), picked with seed 42. The same questions are used in every condition. |
| Models | Qwen3-4B-Instruct-2507 and Phi-4-mini-instruct, float16, no fine-tuning |
| Answers | One main answer (greedy) and 5 sampled answers (temperature 0.7). Format: `Answer: <letter>` and `Reason: <one sentence>` |
| Condition A (`no_ctx`) | Question and options only |
| Condition B (`ctx`) | Question, options and the top 3 retrieved passages |
| Retrieval | BM25 (`bm25s`) over the first 5 million lines of the ARC Corpus. Query = question + all options |
| D0 | 1 minus the share of the 5 sampled answers that match the main answer |
| D1 | 1 minus the mean cosine similarity between the main reason and the 5 sampled reasons (all-MiniLM-L6-v2 embeddings) |
| Metrics | Accuracy, AUROC with 95% bootstrap confidence intervals, paired bootstrap differences, sample agreement rate, repeated-wrong rate |

The positive class for detection is an incorrect main answer. No decision threshold is tuned.

## Results (test set, 300 questions)

| Model | Setting | Accuracy | D0 AUROC | D1 AUROC |
|---|---|---|---|---|
| Qwen3-4B | no context | 0.893 | 0.530 | 0.594 |
| Qwen3-4B | with context | 0.880 | 0.514 | 0.580 |
| Phi-4-mini | no context | 0.820 | 0.642 | 0.655 |
| Phi-4-mini | with context | 0.797 | 0.701 | 0.629 |

Confidence intervals and dev set results are in `data/summary.csv`. An interactive view is on the dashboard.

### Findings

- **RQ1:** retrieved context did not improve accuracy for either model. The small drops are within a few questions.
- **RQ2:** neither detector was clearly better. Every paired difference (D1 minus D0, and context minus no context) has a 95% confidence interval that includes 0. Qwen3-4B gives D0 close to chance; Phi-4-mini gives a modest signal (AUROC about 0.63 to 0.70).
- **RQ3:** Qwen3-4B is almost always self-consistent. With context, all 5 samples match the main answer for 99.7% of test questions, and for 97.2% of its wrong answers all 5 samples repeat the same wrong option, so D0 cannot detect them. For Phi-4-mini with context the repeated-wrong share is 49.2%.

### Retrieval quality

The text of the correct option appeared (exact match) in the retrieved passages for 11.3% of dev questions (17 of 150) and 10.7% of test questions (32 of 300). Retrieval from the ARC Corpus therefore rarely contained the answer text, which may help explain why context did not improve accuracy. Exact matching is strict, so the real coverage may be somewhat higher.

## Limitations

- Only 32 to 61 wrong answers per setting, so confidence intervals are wide and small differences may not be meaningful.
- Two models, one dataset (English, multiple choice, grade-school science).
- A model that repeats the same wrong answer produces low disagreement, so D0 and D1 cannot detect it.
- Fixed sampling (5 samples, temperature 0.7). Retrieval is lexical (BM25) over a subset of the corpus and was not optimised.
- D1 depends on the embedding model and the short explanations, and is treated as a heuristic signal.

## Project documents

- [Project synopsis](docs/synopsis.pdf): problem statement, objectives, literature review and planned method.

## Notebooks

Run in this order in Google Colab. The generation cells need a T4 GPU runtime.

| Notebook | What it does |
|---|---|
| `notebooks/01_no_context_baseline_and_detectors.ipynb` | Creates the fixed question sets, generates answers without context, evaluates D0 and D1 |
| `notebooks/02_retrieval_of_passages.ipynb` | Builds the BM25 index, compares query styles, saves the retrieved passages |
| `notebooks/03_with_context_generation_and_evaluation.ipynb` | Generates answers with context, evaluates D0 and D1, compares both conditions, builds `summary.csv` |

## How to reproduce

1. Create a folder `hallu_project` in Google Drive and put the ARC Corpus zip (`ARC-V1-Feb2018.zip`) in `hallu_project/data/`.
2. In Colab, add your Hugging Face token as a secret named `HF_TOKEN`.
3. Run notebooks 01, 02, 03 in order. Results are saved to `hallu_project` after every question, so a run can be resumed if Colab disconnects.

## Data files (`data/`)

| File | Content |
|---|---|
| `question_sets.json` | Ids of the 150 dev and 300 test questions |
| `questions_resolved.json` | Question text, options and correct answer for those ids |
| `retrieved_dev.json`, `retrieved_test.json` | Top 3 retrieved passages per question |
| `retrieval_hits_dev.json`, `retrieval_hits_test.json` | Whether the correct option text appears in the passages |
| `gen_<model>_<condition>_<split>.jsonl` | Raw generations: main answer, 5 sampled answers, predicted letters and timing per question |
| `scores_<model>_<condition>_<split>.csv` | Per-question D0 and D1 scores |
| `summary.csv` | All accuracies and AUROCs with confidence intervals |
| `run_info.jsonl` | Software versions and GPU used for the runs |

## Datasets, models and tools

- **ARC-Challenge:** https://huggingface.co/datasets/allenai/ai2_arc
- **ARC Corpus (ARC-V1-Feb2018):** https://allenai.org/data/arc
- **Qwen3-4B-Instruct-2507:** https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- **Phi-4-mini-instruct:** https://huggingface.co/microsoft/Phi-4-mini-instruct
- **all-MiniLM-L6-v2:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **bm25s:** https://github.com/xhluca/bm25s

The datasets and models each have their own license. Please check their pages before reuse. The study uses a public benchmark and involves no human participants or personal data.

## References

- Clark et al. (2018). Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401
- Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. arXiv:2303.08896
- Kuhn et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. arXiv:2302.09664
- Reimers and Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT Networks. arXiv:1908.10084
- Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4)

The full literature review is in the project report.

## Author

**Name:** Sejal Singh
**Email:** [sejalsingh8647@gmail.com](mailto:sejalsingh8647@gmail.com)
**LinkedIn:** [Connect with me](https://www.linkedin.com/in/sejal-singh-97669b314)
