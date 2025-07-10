/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.ensemblelearn;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;

import java.util.Random;


public class EnsembleLearn {

    public static void main(String[] args) throws Exception {
        // ARFFファイルの読み込み
        DataSource source = new DataSource("iris.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        int flg = 2;    // 1 : Bagging, 2 : Boosting
        Evaluation eval = null;
        if(flg == 1){
            // Baggingの設定
            Bagging bagger = new Bagging();
            bagger.setClassifier(new RandomTree());
            bagger.setNumIterations(100); // バギングの反復回数を設定

            // モデルの学習
            bagger.buildClassifier(data);

            // 10分割交差検証の実行
            eval = new Evaluation(data);
            eval.crossValidateModel(bagger, data, 10, new Random(1));
            System.out.println("Bagging ==== \n");
        }else if(flg == 2){
            // Boostingの設定
            AdaBoostM1 booster = new AdaBoostM1();
            booster.setClassifier(new DecisionStump());
            booster.setNumIterations(100); // ブースティングの反復回数を設定

            // モデルの学習
            booster.buildClassifier(data);

            // 10分割交差検証の実行
            eval = new Evaluation(data);
            eval.crossValidateModel(booster, data, 10, new Random(1));
            System.out.println("Adaboost ==== \n");
        }
        // 結果の表示
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
