import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "scraim-effort-estimation",
    version = "0.0.1",
    author = "Daniel Veloso",
    author_email = "danielveloso@gmail.com",
    description = "Scraim task effort and duration estimation through machine learning models",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/promessa-project/effort-estimation",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    packages = setuptools.find_packages(),
    install_requires = [
        "category_encoders==2.3.0",
        "Flask==1.1.1",
        "Flask_Cors==3.0.10",
        "googletrans==4.0.0rc1",
        "joblib==0.14.1",
        "matplotlib==3.1.3",
        "nltk==3.4.5",
        "numpy==1.18.1",
        "pandas==1.0.3",
        "PasteScript==3.2.1",
        "scikit_learn==1.0.1",
        "seaborn==0.10.0",
        "setuptools==41.2.0",
        "tabulate==0.8.9",
        "waitress==2.0.0",
        "xgboost==1.4.2",
    ],
    python_requires = ">=3.6"
)