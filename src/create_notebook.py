import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis for QKD Eavesdropper Detection\n",
    "This notebook performs the initial data exploration on the Quantum Key Distribution (QKD) dataset. We will analyze the features, check for anomalies, and prepare the dataset for the Autoencoder and XGBoost models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "%matplotlib inline\n",
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/raw/qkd_dataset_randomized.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Target Variable Distribution\n",
    "Let's look at the distribution of the `Label` column, which indicates whether the record is normal or under attack."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12, 6))\n",
    "sns.countplot(data=df, y='Label', order=df['Label'].value_counts().index, palette='viridis')\n",
    "plt.title('Distribution of QKD Attacks')\n",
    "plt.xlabel('Count')\n",
    "plt.ylabel('Attack Type (Label)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Correlation Matrix\n",
    "Understanding the linear relationships between features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 8))\n",
    "corr = df.drop(columns=['Label']).corr()\n",
    "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')\n",
    "plt.title('Feature Correlation Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Distributions by Attack Type\n",
    "Visualizing how specific features differ between normal traffic and attacks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = df.columns[:-1]\n",
    "plt.figure(figsize=(15, 20))\n",
    "for i, feature in enumerate(features):\n",
    "    plt.subplot(5, 2, i+1)\n",
    "    sns.boxplot(data=df, x='Label', y=feature)\n",
    "    plt.xticks(rotation=90)\n",
    "    plt.title(f'Distribution of {feature} by Label')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion of EDA\n",
    "Based on this EDA, we will next move to feature engineering, where we encode the categorical `Label` column, standardize the numerical features, and build our proposed Autoencoder + XGBoost models."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('../notebooks/01_EDA.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
