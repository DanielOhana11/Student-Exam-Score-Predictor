# Student Exam Score Predictor 🎓📊

A Machine Learning regression project designed to predict students' final exam scores based on various academic and behavioral features. The model achieves high accuracy by applying advanced feature engineering, data transformations, and statistical pruning.

## 🏆 Project Highlights
* **High Accuracy:** Achieved an **R² score of 0.9215** on test data.
* **Robustness:** Demonstrated excellent generalization with a Train R² of 0.9247 (no overfitting).
* **Analytical Approach:** Leveraged log transformations for skewed data and polynomial features to capture non-linear relationships (e.g., the diminishing returns of sleep).

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## 🧠 Methodology & Data Processing

### 1. Data Cleaning
* **Numerical Values:** Missing values were imputed using the **median** to ensure robustness against outliers.
* **Categorical Values:** Missing binary/categorical values were imputed using the **mode**.

### 2. Feature Engineering
* **Log Transformations:** Applied to `study_hours_per_week` and `time_on_social_media_min_per_day` to correct right-skewed (long-tail) distributions, bringing them closer to a normal distribution and improving linear model assumptions.
* **Capturing Non-Linearity (Quadratic Features):**
  * `Study Hours Squared`: Added to capture the exponential reward of studying (studying a lot yields a disproportionately higher score).
  * `Sleep Hours Squared`: Added to model the diminishing returns of sleep. The model successfully identified an optimal sleep threshold (7-8 hours), after which the contribution to the score stops or decreases.

### 3. Feature Selection & Pruning
* **Correlation Analysis:** Removed features with near-zero correlation to the target variable (e.g., commute time, part-time job, favorite music genre, gender) to simplify the model and prevent noise.
* **Outlier Removal:** Conducted Residual Analysis and removed 229 extreme outliers (> 3 standard deviations from the predicted value) to stabilize the model and prevent skewing by unrepresentative noise.

## 📈 Results & Evaluation

The model's predictions align closely with actual scores across the entire distribution range. Residual analysis confirms that the errors are normally distributed around zero with no hidden patterns (U-shapes), validating the linearity and statistical soundness of the engineered features.

**Final Equation Coefficients (Impact on Score):**
* **Positive Impact:** Study Hours (Exponential), Midterm Score, Attendance Rate, Homework Average.
* **Negative Impact:** Social Media Time.
* **Complex Impact:** Sleep Hours (Optimal range yields positive impact, excess/lack yields negative).

## 📁 Repository Structure
* `data/`: Contains the sample training and test datasets.
* `src/`: Contains the Python source code / Jupyter Notebook for data processing and model training.
* `docs/`: Contains the full analytical report detailing the statistical decisions (PDF).
* `results/`: Contains the generated predictions for the test dataset.

---
*Developed by Daniel Ohana.*