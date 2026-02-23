# 🚀 AutoMeta: Adaptive Meta-Learning Framework for Dynamic Task Shifts

AutoMeta is an advanced meta-learning framework designed to handle **dynamic task distribution shifts** in medical image classification.  
The system integrates **Bayesian Online Change Point Detection (BOCPD)**, a **memory bank mechanism**, and **contrastive learning** on top of an extended **MAML (Model-Agnostic Meta-Learning)** architecture.

This project focuses on improving robustness in non-stationary environments where tasks evolve over time.

---

## 📌 Problem Statement

Traditional meta-learning approaches assume static task distributions.  
However, in real-world medical applications:

- Data distributions shift over time
- New classes or domains emerge
- Models must adapt quickly without catastrophic forgetting

AutoMeta addresses these challenges using adaptive task shift detection and memory-augmented learning.

---

## 🧠 Key Innovations

- 🔍 **Task Shift Detection** using Bayesian Online Change Point Detection (BOCPD)
- 🧠 **Memory Bank** for retaining previously learned task knowledge
- 🔄 **Contrastive Learning** for improved feature generalization
- ⚡ **Extended MAML Framework** for fast adaptation
- 📈 Robust learning under dynamic and non-stationary environments

---

## 🏗️ Project Architecture

```
AutoMeta/
│
├── models/              # Meta-learning and backbone model architectures
├── src/                 # Training, evaluation, and experiment scripts
├── utils/               # Helper functions and utilities
├── requirements.txt     # Project dependencies
└── README.md            # Documentation
```

---

## ⚙️ Tech Stack

- Python
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/anikett0307/AutoMeta.git
cd AutoMeta
```

Create virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the model:

```bash
python src/train.py
```

Evaluate:

```bash
python src/evaluate.py
```

---

## 📊 Objective

To build a meta-learning system capable of:

- Detecting task distribution shifts in real-time
- Adapting quickly to new tasks
- Reducing catastrophic forgetting
- Maintaining strong generalization performance in medical imaging tasks

---

## 📈 Future Improvements

- Add real-world medical datasets integration
- Deploy with Streamlit demo interface
- Add visualization dashboard
- Convert framework into research publication format
- Implement continual learning benchmarks

---

## 💡 Applications

- Medical image classification
- Continual learning systems
- Adaptive AI systems
- Dynamic domain adaptation
- Real-time learning systems

---

## 👨‍💻 Author

**Aniket M H**  
B.E Computer Science  
Aspiring AI/ML Engineer  

GitHub: https://github.com/anikett0307

---

## ⭐ If You Found This Project Useful

Consider giving it a star ⭐ on GitHub!
