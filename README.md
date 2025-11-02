# Sentiment Analysis on Real-World E-Commerce Dataset (Amazon & Flipkart)

## 📖 Description

This project tackles the problem of **multi-class sentiment analysis** for e-commerce product reviews. In today's data-driven world, understanding customer feedback is crucial for businesses to gauge product reception, identify issues, and make informed decisions. This system moves beyond simple positive/negative classification to predict a specific 1-to-5 star rating based on the review text.

We explore and compare five different machine learning and deep learning models to determine the most effective approach. The project culminates in a fully interactive **Streamlit web application** where users can upload data, train models, and perform live predictions on new reviews.

A key part of this analysis was exploring **cross-domain generalization** (training on Amazon data, testing on Flipkart data) and handling severe **class imbalance** through oversampling and hyperparameter tuning to achieve high-performance, robust models.

### ✨ Key Features of the Project

* **📊 Exploratory Data Analysis (EDA):** Upload a dataset to see an interactive overview, including rating distributions, review length analysis, and brand breakdowns.
* **⚙️ Multi-Model Training:** Train five different models (Naive Bayes, Logistic Regression, SVM, Random Forest, LSTM) on the fly with configurable parameters (test size, TF-IDF features).
* **🔮 Live Prediction:**
    * **Single Review:** Type or paste a new review for an instant 1-5 star rating prediction.
    * **Batch Upload:** Upload a CSV of new reviews and get predictions for all of them, with an option to download the results.
* **📈 Results & Metrics:** Get a detailed performance breakdown for each trained model, including:
    * Accuracy, F1-Score, Precision, and Recall.
    * Interactive Confusion Matrices.
    * Detailed Classification Reports.
    * A comparative radar chart showing all model metrics at a glance.

---

🔗 **Live Deployed Project:** [https://shreyas-sentiment-analysis-on-real-world-e-commerce-dataset.streamlit.app](https://shreyas-sentiment-analysis-on-real-world-e-commerce-dataset.streamlit.app/)

### 🎥 Project Demonstration

https://github.com/user-attachments/assets/3248b04b-f1c2-4819-b8e7-46287d07b826

---

## 💾 Dataset

### Source
The initial dataset was a composite of product reviews from various e-commerce sites (primarily Amazon and Flipkart) sourced from Kaggle, data.world, and UCSD.

* **Kaggle Dataset Link:** [E-Commerce Product Review Data](https://www.kaggle.com/datasets/vivekgediya/ecommerce-product-review-data/data)

### Initial Data
The raw dataset (`Product Review Large Data.csv`) contained **10,971 entries** and 27 columns. A preliminary analysis showed a significant class and source imbalance:
* **Brand:** Flipkart (9,374 reviews), Amazon (1,585 reviews)
* **Ratings:** Heavily skewed towards 5-star reviews.

### Preprocessing and Enhancement
The data underwent a two-phase enhancement process:

1.  **Phase 1: Initial Cleaning & Splitting**
    * Dropped all columns except `reviews.text`, `reviews.rating`, and `brand`.
    * Removed rows with missing values.
    * Separated the data into two dataframes (Amazon and Flipkart) to analyze cross-domain performance.
    * Applied a text cleaning function:
        * Converted text to lowercase.
        * Removed HTML tags, URLs, and non-alphabetic characters.
        * Removed standard English stopwords.
        * Applied lemmatization to reduce words to their root form.

2.  **Phase 2: Handling Class Imbalance (Hyperparameter Tuning Dataset)**
    * To build the final, robust models for the app, the class imbalance was addressed.
    * The `reviews.rating` column was found to be heavily skewed.
    * **RandomOverSampler** (from `imbalanced-learn`) was used to oversample the minority classes (1, 2, 3, and 4-star reviews) to match the number of 5-star reviews.
    * This resulted in a new, balanced dataset (`Enhanced_Product_Review_Data.csv`) with **23,725 entries**, used for final model tuning and evaluation.

---

## 🛠️ Methodology

The problem was framed as a **multi-class text classification** task. The core challenge was transforming raw text into numerical features that models can understand and then comparing different modeling architectures.

### 1. Feature Engineering
Two distinct paths were taken for feature engineering based on the model type:

* **Path A: TF-IDF (for Classical ML Models)**
    * **Technique:** Term Frequency-Inverse Document Frequency (`TfidfVectorizer`).
    * **Why:** This approach weighs words based on their importance in a document relative to the entire corpus. It's highly effective for classical ML models and captures which words are most discriminative for a particular rating.
    * **Hyperparameter:** A `max_features` limit of 5,000 was used to keep the feature space manageable and filter out extremely rare words.

* **Path B: Tokenized Sequences (for Deep Learning)**
    * **Technique:** `Tokenizer` and `pad_sequences` from Keras.
    * **Why:** LSTMs require sequences of integers as input, where each integer represents a word in a vocabulary. This method preserves the *order* of words, which is critical for LSTMs to learn context and sequential patterns.
    * **Hyperparameters:**
        * `vocab_size`: 10,000 (Top 10,000 most frequent words).
        * `max_length`: 150 (Reviews are padded or truncated to this length).
* **Alternatives Considered:** We considered using pre-trained embeddings like Word2Vec or GloVe. However, we opted to train our own `Embedding` layer from scratch. This allows the model to learn word vector representations that are highly specific to the *nuances and vocabulary of e-commerce reviews*, which may not be well-represented in generic pre-trained models.

### 2. Model Architecture
Five models were trained and evaluated to compare performance:

1.  **Multinomial Naive Bayes (NB):** A probabilistic baseline model that is fast and performs well on text classification tasks, assuming conditional independence between features.
2.  **Logistic Regression (LR):** A robust and interpretable linear model that is highly effective for text classification, especially with TF-IDF features.
3.  **Linear Support Vector Machine (SVM):** A powerful linear classifier that works by finding the optimal hyperplane to separate classes. It is known for its high performance on high-dimensional sparse data like TF-IDF vectors.
4.  **Random Forest (RF):** An ensemble model that builds multiple decision trees and merges their results. It's robust to overfitting and can capture complex non-linear relationships.
5.  **LSTM (Deep Learning):** A Long Short-Term Memory network, which is a type of Recurrent Neural Network (RNN). It's designed to learn long-range dependencies and sequential patterns in data, making it theoretically ideal for text.

* **Why this approach?** This selection allows us to compare computationally efficient classical models against a more complex sequential model. This helps answer a key question: *Is the added complexity and training time of an RNN necessary for this task, or can a tuned classical model like Random Forest or SVM achieve superior results?*
* **Alternatives Considered:** More advanced (and computationally expensive) transformer models like BERT or RoBERTa were considered. However, the chosen models provide a strong and practical baseline, which is the focus of this project.
---

## 🚀 Steps to Run the Code

To run the interactive Streamlit application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_LINK]
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    *Create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`.*
    ```
    streamlit
    pandas
    numpy
    plotly
    textblob
    scikit-learn
    tensorflow
    nltk
    imbalanced-learn
    ```

4.  **Download NLTK data:**
    Run Python and in the interpreter, type:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    ```

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

6.  **Open the app in your browser** at `http://localhost:8501`.

---

## 🔬 Experiments & Results

### Hyperparameter Tuning
Before final evaluation, the four classical ML models were tuned using `GridSearchCV` on the balanced dataset (`Enhanced_Product_Review_Data.csv`) to find their optimal parameters.

* **Naive Bayes:** `{'alpha': 0.1}`
* **Logistic Regression:** `{'C': 10, 'multi_class': 'auto', 'solver': 'lbfgs'}`
* **SVM (LinearSVC):** `{'C': 10, 'loss': 'squared_hinge'}`
* **Random Forest:** `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}`

### Results Summary
The tuned models were evaluated on the balanced test set. **Random Forest emerged as the top-performing model** by a significant margin, achieving a 95% F1-score. This demonstrates that with proper preprocessing, feature engineering, and tuning, ensemble methods can outperform baseline deep learning models on this task.

| Model | Dataset | Accuracy | F1-Score |
|:---|:---|---:|---:|
| Naive Bayes | Tuned Test Set | 0.7353 | 0.7346 |
| Logistic Regression | Tuned Test Set | 0.8411 | 0.8394 |
| SVM | Tuned Test Set | 0.8468 | 0.8447 |
| **Random Forest** | **Tuned Test Set** | **0.9498** | **0.9496** |
| LSTM (Baseline) | Tuned Test Set | 0.6738 | 0.5725 |

### Comparison with Published Methods

To contextualize this project's results, we compare our best model's performance against benchmarks from published research on similar Amazon 1-5 star review classification tasks.

| Model | Performance (Accuracy / F1) | Context / Source |
|:---|:---|:---|
| **This Project (Tuned Random Forest)** | **94% F1-Score** | **Our enhanced, balanced dataset** |
| BERT (State-of-the-Art) | 85-93% Accuracy | Comparative study of 9 models |
| BERT (State-of-the-Art) | 89% Accuracy | Comparative study on 400k Amazon reviews |
| RateNet (1D CNN + BiGRU) | 86.6% Accuracy | Hybrid DL model on Amazon 5-core dataset |
| SVM (Tuned) | 80.8% - 81.9% Accuracy | Published benchmarks on Amazon reviews |
| BERT (fine-tuned) | 80.0% Accuracy | Hugging Face model for 1-5 star classification |

**Analysis:** Our tuned **Random Forest** model, at **95% F1-score**, performs exceptionally well. It not only surpasses other classical baselines like SVM but also outperforms several state-of-the-art transformer-based **BERT** models from published benchmarks (which scored between 85-93%).

This suggests that for this specific dataset, the combination of **effective class balancing** (using `RandomOverSampler`) and **thorough hyperparameter tuning** was more impactful than model architecture alone. It highlights that a well-engineered classical ensemble model can be a highly effective and computationally cheaper alternative to more complex deep learning architectures.

### Cross-Domain Generalization Gap
An initial experiment was conducted (in the notebook) to test generalization. Models were trained *only* on Amazon data and then tested on both an Amazon validation set (in-domain) and the Flipkart dataset (cross-domain).

This revealed a significant **generalization gap**, where performance dropped substantially when moving from Amazon to Flipkart reviews, highlighting the different language, slang, and context used on the two platforms.

| model | Amazon (Cross-Domain Test) | Amazon (In-Domain Validation) | Flipkart (Cross-Domain Test) | Performance Drop (Generalization Gap) - Amazon | Performance Drop (Generalization Gap) - Flipkart |
|:---|---:|---:|---:|---:|---:|
| Random Forest | 0.9358 | 0.6097 | 0.4249 | -0.3261 | 0.1848 |
| SVM | 0.9265 | 0.6307 | 0.4763 | -0.2957 | 0.1545 |
| Logistic Regression | 0.7042 | 0.6100 | 0.4242 | -0.0942 | 0.1857 |
| LSTM | 0.6573 | 0.5683 | 0.4205 | -0.0890 | 0.1479 |
| Naive Bayes | 0.5969 | 0.5725 | 0.4205 | -0.0245 | 0.1520 |

*(Note: The negative drop for Amazon indicates the test set was easier than the validation set, likely due to the oversampling and splitting logic in that specific experiment. The key metric is the **Flipkart gap**, which shows a ~15-18% F1-score drop across all models.)*

---

## 🏁 Conclusion

This project successfully demonstrates the end-to-end process of building a high-performance sentiment analysis model.

* **Key Result:** A tuned **Random Forest** classifier, trained on a balanced dataset with TF-IDF features, was the most effective model, achieving an F1-score of **0.9496** for 5-class rating prediction.
* **Key Learning:** Data preprocessing and, most importantly, **handling class imbalance** (via RandomOverSampler) were the most critical steps for improving model performance. The baseline models performed poorly on the initial imbalanced data.
* **Domain Challenge:** The initial cross-domain analysis confirmed that models trained on one e-commerce platform (Amazon) do not generalize well to another (Flipkart) without fine-tuning, due to differences in review style and vocabulary.
* **Application:** The final Streamlit app provides a practical and user-friendly interface for anyone to leverage these trained models for live predictions, demonstrating the real-world applicability of the project.

---

## 📚 References

* Pang, B., Lee, L., and Vaithyanathan, S. (2002). [Thumbs up? Sentiment Classification using Machine Learning Techniques](https://aclanthology.org/W02-1011.pdf). In *Proceedings of EMNLP*.
* Kaggle Dataset: [E-Commerce Product Review Data](https://www.kaggle.com/datasets/vivekgediya/ecommerce-product-review-data/data)
* UCSD Web Mining Lab: [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)
* data.world: [Amazon and Flipkart Review Datasets](https://data.world/community/datasets?q=amazon+flipkart+reviews)
