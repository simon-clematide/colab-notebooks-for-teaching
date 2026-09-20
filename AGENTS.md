# Agent Instructions for `colab-notebooks-for-teaching`

This repository contains a curated collection of educational Jupyter/Colab notebooks used for teaching Machine Learning and Natural Language Processing at the University of Zurich (Institute of Computational Linguistics / Prof. Simon Clematide), notably supporting:
- **ML4NLP 1** (Machine Learning for Natural Language Processing 1)
- **Text Mining / Computational Linguistics**
- Specialized research and seminar topics (Historical NLP / Impresso Project, Agentic RAG, Frontier LLM APIs)

Use this guide as the technical standard and map when adding, modifying, refactoring, or verifying notebooks in this repository.

---

## 1. Core Pedagogical Philosophy

1. **Zero Friction in Google Colab:**
   Students must be able to click the "Open in Colab" badge and run the code immediately without complex manual setup, broken dependencies, or local environment configuration.
2. **Pedagogical Progression (First Principles to High-Level Abstractions):**
   Where relevant, start with fundamental mechanics (e.g. raw tensors, manual forward passes, explicit autodiff `loss.backward()`, parameter updates) before introducing black-box modules (`nn.Module`, high-level pipelines).
3. **Visual Intuition over Raw Numbers:**
   Avoid dumping large floating-point arrays or endless text logs. Always pair mathematical concepts with clear visualizations (e.g., 2D decision boundaries, loss trajectories, side-by-side hidden space coordinate transformations, embedding scatter plots).
4. **Clean, Modular, Reproducible Code:**
   - Always set explicit random seeds for reproducibility (`torch.manual_seed(seed)`, `np.random.seed(seed)`).
   - Eliminate antipatterns like mutating global variables across different notebook cells.
   - Encapsulate reusable helper routines (like plotting functions) cleanly.

---

## 2. Notebook Catalog & Curricular Areas

| Notebook | Primary Frameworks | Curricular Topic / Role | Notes / Status |
| :--- | :--- | :--- | :--- |
| **`BabyGradient.ipynb`** | SymPy, NumPy | Mathematical optimization; symbolic vs numerical gradients | Introductory optimization |
| **`Reverse_Autodiff_Example_in_Tensorflow_and_Pytorch.ipynb`** | PyTorch, TensorFlow | Computational graphs; reverse-mode automatic differentiation | Direct comparative study |
| **`pytorch_basics.ipynb`** | PyTorch | PyTorch tensors, slicing, broadcasting, device management | First-contact PyTorch tutorial |
| **`pytorch_autograd_matrix.ipynb`** | PyTorch | Matrix-level autograd, vector-Jacobian products (VJP) | Mathematical foundations of autodiff |
| **`Pragmatic_Gradients.ipynb`** | PyTorch | Optimization loops, gradient steps, manual vs optimizer updates | Practical autodiff mechanics |
| **`xor_Logistic_Regression_With_PyTorch.ipynb`** | PyTorch, Matplotlib | The XOR problem: failure of linearity, feature engineering ($x_1 \cdot x_2$), MLP representation learning | **Current primary XOR notebook** (includes hidden layer space visualization) |
| **`xor-handcrafted-ffnn.ipynb`** | NumPy | Handcrafted weights in a 2-layer network solving XOR | Conceptual precursor to backprop |
| **`xor_Logistic_Regression_With_Autograd.ipynb`** | HIPS `autograd` | Legacy XOR logistic regression | **Deprecated** (superseded by PyTorch version) |
| **`sklearn_text_classification.ipynb`** / **`text_classification.ipynb`** | scikit-learn | TF-IDF vectorization, Logistic Regression, Naive Bayes | Core classical NLP classification pipeline |
| **`plot_document_clustering.ipynb`** | scikit-learn | Unsupervised document clustering (K-Means, evaluation) | Clustering and text representation |
| **`notebooks/topic_modeling_sklearn.ipynb`** | scikit-learn, pyLDAvis | Topic modeling (LDA, NMF) on 20 Newsgroups with interactive visualization | Topic models |
| **`word2vec_tensorflow.ipynb`** | TensorFlow | Continuous Skip-Gram model trained from scratch | Word representations from first principles |
| **`openai_word_embeddings.ipynb`** / **`OpenAI_Embeddings_Visualization_Notebook.ipynb`** | OpenAI API, scikit-learn | Subword embeddings, multilingual geometry, dimensionality reduction | Modern dense representations |
| **`BERT_for_Sentiment_Analysis_SC_FS26.ipynb`** | Hugging Face Transformers, PyTorch | Fine-tuning pretrained BERT for binary sentiment analysis | Core deep learning NLP paradigm |
| **`sentiment-analysis-overview.ipynb`** / **`sentiment_analysis_overview.ipynb`** | NLTK, TextBlob, VADER, Flair, Transformers | Broad comparative benchmark of rule-based vs neural sentiment analyzers | Comprehensive overview |
| **`sentiment_emotion_analysis.ipynb`** | NRCLex, Flair | Fine-grained emotion and sentiment classification | Lexicon vs neural comparison |
| **`flair_verb_frame_tagging.ipynb`** | Flair | FrameNet semantic role labeling and verb frame extraction | Semantic parsing |
| **`impresso_ner_displacy_visualization.ipynb`** | spaCy, displaCy | Named Entity Recognition (NER) visual analysis on historical newspapers | Impresso Project pipeline |
| **`impresso_nel_minimal.ipynb`** / **`MGENRE_impresso_linking.ipynb`** | Hugging Face, mGENRE | Multilingual Named Entity Linking (NEL) using sequence-to-sequence models | Entity disambiguation |
| **`floret-language-identification.ipynb`** | floret, Hugging Face | Compact subword-based language identification | FastText/floret compact embeddings |
| **`g2p.ipynb`** | PyTorch / Seq2Seq | Neural grapheme-to-phoneme transduction | Sequence transduction |
| **`openai-keyphrase-starter.ipynb`** / **`openai_image_description.ipynb`** | OpenAI API | Prompt engineering, structured extraction, multimodal prompting | LLM / VLM API interaction |
| **`reasoning-model-conversation.ipynb`** | Transformers, PyTorch | Structured dialog extraction and serialization for reasoning models | Conversation formatting |
| **`multi-document-agentic-rag_main.ipynb`** | LlamaIndex | Multi-document Agentic Retrieval-Augmented Generation (RAG) | Advanced agentic architectures |
| **`colab-ai-text-mining.ipynb`** | Google GenAI SDK | AI-assisted text mining workflows | Applied GenAI |

---

## 3. Authoring & Editing Standards for Notebooks

### 3.1 Colab Badge & Header Metadata

Every student-facing notebook must have:
1. The standard **Open in Colab** badge as the first markdown cell:
   ```html
   <a href="https://colab.research.google.com/github/simon-clematide/colab-notebooks-for-teaching/blob/main/<Notebook_Name>.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
   ```
2. Correct notebook metadata in the root JSON:
   ```json
   "metadata": {
     "colab": {
       "provenance": [],
       "include_colab_link": true
     },
     "kernelspec": {
       "display_name": "Python 3",
       "name": "python3"
     },
     "language_info": {
       "name": "python"
     }
   }
   ```
3. A clear H1 title (`# Title`) and a summary section outlining the learning goals and syllabus context.

### 3.2 Package Installation & Setup Cells

- Prefer packages preinstalled in Google Colab (PyTorch, torchvision, scikit-learn, matplotlib, pandas, numpy, scipy).
- If additional libraries are required (e.g. `flair`, `transformers`, `openai`, `llama-index`, `pyLDAvis`), place an explicit, quiet install cell right after the introductory markdown:
  ```python
  # Run this cell if executing in Google Colab
  import sys
  if "google.colab" in sys.modules:
      !pip install -q transformers datasets evaluate
  ```
- Guard against interactive widgets or GUI blockers that freeze headless execution or non-Colab Jupyter environments.

### 3.3 Hardware Acceleration & Device Neutrality

Always support both CPU and GPU seamlessly:
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
Ensure that models and tensors are explicitly routed to `device`, but small educational examples (like XOR, 2-layer MLPs, or small NumPy demonstrations) should run comfortably on CPU without requiring a GPU quota.

### 3.4 Data Access & Self-Containment

- Small benchmark datasets should reside in the repository's `data/` folder (e.g., `data/complaint1700.csv`, `data/impresso-lid.tsv`).
- When a notebook depends on a file in `data/`, provide a resilient loading helper that works both locally and in Google Colab:
  ```python
  import os

  DATA_PATH = "data/complaint1700.csv"
  if not os.path.exists(DATA_PATH) and "google.colab" in sys.modules:
      !git clone --depth 1 https://github.com/simon-clematide/colab-notebooks-for-teaching.git repo
      DATA_PATH = "repo/data/complaint1700.csv"
  ```
- Do not rely on Colab's default temporary `sample_data/` folder for course exercises.

---

## 4. Local Execution, Testing, and Quality Assurance

### 4.1 Running and Testing Notebooks Headless

We use `uv` for isolated, fast local execution without requiring a global Python virtualenv.

To execute a notebook end-to-end and pre-render all plots, tables, and execution outputs:
```bash
uv run --with nbconvert --with torch --with matplotlib --with scikit-learn \
  jupyter nbconvert --to notebook --execute --inplace <Notebook_Name>.ipynb
```

### 4.2 Validating JSON Integrity

Before committing any notebook:
1. Verify the JSON structure parses without syntax errors:
   ```bash
   python3 -m json.tool <Notebook_Name>.ipynb > /dev/null
   ```
2. Verify that `cells` have valid types (`markdown`, `code`) and valid string/list fields.
3. Check `git diff` to ensure that outputs do not introduce massive binary noise or megabytes of redundant checkpoint state.

---

## 5. Cross-Repository Relationships

This repository is directly connected to the lecture slide decks in the teaching repository `ml1script` (`/Users/siclemat/lehre/hs26/ml4nlp1/ml1script/`):

- Slide decks in `ml1script` link to notebooks in this repo via GitHub and Google Colab URLs:
  - `ml-and-linear-classification.tex` / `linear-classification.tex`
  - `auto-diff.tex`
  - `pytorch.tex`
  - `ffnn.tex`
- **Link Stability Rule:** If you rename, reorganize, or deprecate a notebook in this repository, check and update the corresponding `\href[*]{...}` links in `ml1script`.
- **Prefer PyTorch over HIPS Autograd:** Legacy course links to `xor_Logistic_Regression_With_Autograd.ipynb` should be updated to point to `xor_Logistic_Regression_With_PyTorch.ipynb`.
