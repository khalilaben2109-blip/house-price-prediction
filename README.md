# 🏠 House Price Prediction - Advanced ML Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

> **🎯 Complete Machine Learning solution for house price prediction with 5 advanced algorithms, interactive web interfaces, and 98.7% accuracy**

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch interactive menu
python start.py

# Or launch web interface directly
streamlit run app.py
```

## ✨ Key Features

### 🤖 **5 Advanced ML Algorithms**
- **Linear Regression** - Fast and interpretable (98.7% accuracy)
- **Random Forest** - Robust ensemble method (98.7% accuracy)  
- **XGBoost** - Gradient boosting champion (98.0% accuracy)
- **Gradient Boosting** - Sequential ensemble (98.6% accuracy)
- **Support Vector Regression** - Non-linear modeling

### 📊 **Diverse Data Sources**
- **Mixed Dataset** - 1,500+ samples with 15 realistic features
- **California Housing** - 20,640 real estate records
- **Synthetic Data** - Generated with realistic pricing formulas
- **Online Data** - Automatic loading from public repositories

### 🌐 **Interactive Web Interfaces**
- **Main Interface** (http://localhost:8501) - ML training and predictions
- **Database Management** (http://localhost:8502) - Data administration
- **Real-time Visualizations** - Plotly interactive charts
- **Custom Predictions** - Slider-based property configuration

### 💾 **SQLite Database Integration**
- **Persistent Storage** - All data, models, and predictions saved
- **Complete History** - Track all training sessions and results
- **Advanced Analytics** - Performance trends and comparisons
- **Export/Import** - CSV data management

## 🏗️ Project Architecture

```
house-price-prediction/
├── 🌐 app.py                     # Main Streamlit interface
├── 🗄️ database_app.py            # Database management interface
├── 🚀 demo_advanced.py           # Advanced 5-model demonstration
├── 💾 demo_database.py           # Database integration demo
├── 🎮 start.py                   # Interactive launcher menu
├── 🎨 showcase.py                # Visual presentation
│
├── 📁 src/                       # Core source code
│   ├── 📊 data/                  # Data management
│   │   ├── data_loader.py        # Multi-source data loading
│   │   ├── data_generator.py     # Synthetic data generation
│   │   └── preprocessor.py       # Data preprocessing pipeline
│   ├── 🤖 models/                # ML algorithms
│   │   ├── linear_regression_model.py
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   ├── gradient_boosting_model.py
│   │   └── support_vector_model.py
│   ├── 📈 evaluation/            # Model evaluation
│   ├── 🎨 visualization/         # Interactive charts
│   ├── ⚙️ optimization/          # Hyperparameter tuning
│   ├── 💾 database/              # SQLite management
│   └── 📄 reports/               # PDF report generation
│
├── 📓 notebooks/                 # Jupyter analysis
├── 🧪 tests/                     # Unit tests
├── ⚙️ config/                    # Configuration files
└── 📊 data/                      # Data storage
```

## 📈 Performance Results

| Model | RMSE | MAE | R² Score | Training Time |
|-------|------|-----|----------|---------------|
| **Random Forest** | 12,596 | 8,129 | **98.7%** | 17.8s |
| **Linear Regression** | 12,906 | 10,040 | **98.7%** | 0.02s |
| **Gradient Boosting** | 12,992 | 8,450 | **98.6%** | 4.4s |
| **XGBoost** | 15,731 | 9,303 | **98.0%** | 0.17s |
| **Support Vector** | 119,145 | 91,886 | -14.5% | 11.4s |

*Results on 1,500 mixed dataset samples*

## 🎯 Usage Examples

### 🖥️ Command Line Interface
```bash
# Interactive menu with all options
python start.py

# Train all 5 models with advanced datasets
python demo_advanced.py

# Database integration demonstration
python demo_database.py

# Generate professional PDF report
python src/reports/report_generator.py
```

### 🌐 Web Interface Usage
```python
# Launch main ML interface
streamlit run app.py

# Launch database management
streamlit run database_app.py --server.port 8502
```

### 🤖 Programmatic Usage
```python
from src.data.data_loader import DataLoader
from src.models.xgboost_model import XGBoostModel

# Load data
loader = DataLoader(data_source='mixed')
X, y = loader.load_boston_housing()

# Train model
model = XGBoostModel()
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step-by-Step Installation
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests to verify installation
python tests/test_models.py

# 5. Launch the application
python start.py
```

## 🎮 Interactive Features

### 🎯 **Custom Predictions**
- Slider-based property configuration
- Real-time price estimation
- Confidence scoring
- Feature importance analysis

### 📊 **Data Exploration**
- Interactive correlation matrices
- Dynamic filtering and sorting
- Statistical summaries
- Distribution visualizations

### 🔧 **Model Management**
- Hyperparameter optimization
- Performance comparison
- Training history tracking
- Model versioning

## 🧪 Testing

```bash
# Run all tests
python tests/test_models.py

# Validate all components
python validation_finale.py

# Performance benchmarking
python demo_advanced.py
```

## 📊 Database Schema

The SQLite database includes:
- **properties** - Real estate data (506+ records)
- **predictions** - Model predictions with timestamps
- **trained_models** - Model performance history
- **training_logs** - Detailed training information

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Streamlit** - Web interface framework
- **Plotly** - Interactive visualizations
- **SQLite** - Database management

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/house-price-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/house-price-prediction/discussions)
- **Documentation**: See `/docs` folder for detailed guides

---

⭐ **Star this repository if you found it helpful!**

🔗 **Live Demo**: [Add your deployed version link here]