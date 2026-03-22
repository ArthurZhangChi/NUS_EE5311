# Battery State-of-Charge (SOC) Estimation

## From Data-Driven Models to Physics-Guided and Probabilistic Learning

This project investigates battery State-of-Charge (SOC) estimation using multiple modelling paradigms, including data-driven, differentiable, scientific machine learning (SciML), and probabilistic approaches.

The goal is to provide a comparative study of these methods under a unified framework.

---

## 📁 Project Structure

```
.
├── datasets/          # Battery datasets (4 datasets used in this project)
├── outputs/           # Experimental results (predictions, metrics, plots)
├── models/            # Model implementations (optional)
├── scripts/           # Training / evaluation scripts (optional)
├── README.md
```

---

## 📊 Datasets

All datasets used in this project are stored in the `datasets/` folder.

* The project uses **four battery datasets**
* Each dataset contains measurements such as:

  * Voltage (V)
  * Current (I)
  * Temperature (T)
  * Ground truth SOC

These datasets are used consistently across all modelling approaches to ensure fair comparison.

---

## ⚙️ Methods

We implement and compare four different approaches:

### 1. Data-Driven Model

* Pure machine learning approach (e.g., MLP)
* Learns SOC directly from input features
* No physical constraints

---

### 2. Differentiable Model

* Based on battery dynamic equations (ODE)
* Uses **automatic differentiation (AD)** to learn parameters
* Example:

  * Learning battery capacity or model parameters

---

### 3. Scientific Machine Learning (SciML)

* Combines physics-based models with neural networks
* Uses **Universal Differential Equations (UDE)**
* Structure:

  ```
  dx/dt = f_physics + f_NN
  ```

---

### 4. Probabilistic Model

* Models uncertainty in SOC estimation
* Outputs:

  * Mean prediction
  * Uncertainty (variance / confidence interval)
* Handles:

  * Measurement noise
  * Battery degradation

---

## 📈 Evaluation Metrics

All models are evaluated using the following metrics:

* **RMSE (Root Mean Square Error)**
* **MAE (Mean Absolute Error)**
* **R² (Coefficient of Determination)**

These metrics are computed consistently across all datasets and models.

---

## 📂 Outputs

All experimental results are stored in the `outputs/` folder.

This includes:

* Predicted SOC curves
* Ground truth comparisons
* Evaluation metrics (RMSE, MAE, R²)
* Visualization plots

---

## 🚀 How to Run

(You can modify this part based on your actual scripts)

Example:

```
python train_baseline.py
python train_differentiable.py
python train_sciml.py
python train_probabilistic.py
```

---

## 📌 Key Idea

This project demonstrates that battery SOC estimation is not just a regression problem, but a **scientific machine learning problem** that benefits from:

* Physics-guided modelling
* Differentiable computation
* Probabilistic uncertainty estimation

---

## 👥 Contributions

* **Yingying Liu** – Data-driven model implementation and documentation
* **Zhenrong Zhan** – Physics-based model implementation and documentation
* **Chi Zhang** – SciML model implementation and documentation
* **Yuansheng Cai** – Probabilistic model implementation and documentation
* **Wendi Yu** – Introduction, background, and comparative analysis

---

## 📄 License

This project is for academic purposes (EE5311 CA1 assignment).
