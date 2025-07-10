# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Computer Simulation course project (CS-9) from Tokyo City University focusing on machine learning experiments using the Weka library. The project implements neural networks and random forests for both classification and regression tasks.

## Commands

### Build and Run
```bash
# Build all modules
cd resources && mvn clean install

# Run individual modules (from resources directory)
# Neural Network Classification
cd nn_weka && mvn exec:java -Dexec.mainClass="com.mycompany.nn_weka.Nn_weka"

# Neural Network Regression  
cd nnRegression && mvn exec:java -Dexec.mainClass="com.mycompany.nnregression.NnRegression"

# Random Forest
cd randomF && mvn exec:java -Dexec.mainClass="com.mycompany.randomf.RandomF"

# Ensemble Learning (Bagging/Boosting)
cd ensembleLearn && mvn exec:java -Dexec.mainClass="com.mycompany.ensemblelearn.EnsembleLearn"
```

### Clean
```bash
# Clean all target directories
cd resources && mvn clean
```

## Architecture

The codebase consists of four independent Maven modules under the `resources/` directory:

1. **nn_weka**: Neural network classification using MultilayerPerceptron
   - Performs grid search over learning rate, momentum, training time, and hidden layers
   - Uses breast-cancer.arff dataset for the assignment
   - Outputs top 5 parameter combinations ranked by accuracy

2. **nnRegression**: Neural network regression 
   - Similar grid search approach but evaluates using RMSE
   - Uses housing.arff dataset
   - Outputs top 5 parameter combinations ranked by RMSE (lower is better)

3. **randomF**: Random forest implementation
   - Supports both classification and regression
   - Must be modified to test different numbers of trees (10, 50, 100, 500, 1000)
   - Should analyze feature importance for regression tasks

4. **ensembleLearn**: Ensemble methods
   - Implements Bagging (flag=1) and AdaBoost (flag=2)
   - Uses RandomTree for Bagging, DecisionStump for AdaBoost
   - Configurable via the `flag` variable in main method

## Key Files and Locations

- Data files: `resources/*/src/main/resources/*.arff`
- Main classes: `resources/*/src/main/java/com/mycompany/*/`
- Assignment submission: `cs-9.md`

## Assignment Requirements

The project must complete four tasks:

1. **Neural Network Classification**: Optimize parameters for breast-cancer.arff
2. **Neural Network Regression**: Optimize parameters for housing.arff  
3. **Random Forest Classification**: Test various tree counts on breast-cancer.arff
4. **Random Forest Regression**: Analyze feature importance on housing.arff

Each task requires running experiments, recording results in the submission document (cs-9.md), and analyzing the findings.

## Important Notes

- All modules use Java 17 and Weka 3.9.6
- Cross-validation (10-fold) is used for all evaluations
- Results should be formatted as tables in the submission document
- The randomF module currently uses iris.arff but must be changed to use the required datasets