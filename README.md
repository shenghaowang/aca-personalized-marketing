# Algorithmic Collective Action for Personalized Marketing

Investigate the impacts of algorithmic collection action on personalized marketing powered by uplift modeling.

## 📦 Installation

While Docker is the recommended method for setting up the environment, the following steps provide a quick alternative using a Python virtual environment:

1. **Clone the repository**
```bash
git clone https://github.com/your-username/text-classification-cnn-lstm.git
cd text-classification-cnn-lstm
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the `pre-commit` hooks**
```bash
pre-commit install
pre-commit run --all-files
```

---

## 🔥 Usage

### Train and export uplift model

```bash
export PYTHONPATH=src
python src/model/main.py data=<dataset_name> model=<model_type>
```

Supported dataset names:

* `starbucks`: Starbucks Dataset
* `hillstrom`: [Kevin Hillstrom Dataset](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)

Supported model types:

* `lgbm`: LightGBM Class Transformation model
* `mlp`: Multi-Layer Perceptron Class Transformation model
* `uplift_rf`: Uplift Random Forest

### Conduct ACA experiments

```bash
export PYTHONPATH=src
python src/experiment/main.py data=<dataset_name> model=<model_type>
```

### Compare Uplift Models with Qini Curves

```bash
export PYTHONPATH=src
python src/model/plot_qini_curves.py data=<dataset_name>
```

### Aggregate Experiment Results

```bash
export PYTHONPATH=src
python src/experiment/agg_results.py data=<dataset_name>
```

### Compare the Distribution of Uplift Scores and Normalized Ranking Before and After the Collective Action

```bash
export PYTHONPATH=src
python src/experiment/uplift_analysis.py data=<dataset_name> model=<model_type>
```
