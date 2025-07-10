package com.mycompany.randomf;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;

public class RandomFClassification {
    public static void main(String[] args) throws Exception {
        // breast-cancer.arffファイルの読み込み
        DataSource source = new DataSource("breast-cancer.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("=== ランダムフォレスト分類実験 (breast-cancer.arff) ===\n");
        System.out.println("木の本数\t精度(%)\t\t正解数/全体数");
        System.out.println("----------------------------------------");

        // 木の本数を変化させて実験
        int[] treeNumbers = {1, 5, 10, 50, 100, 500};
        
        for (int numTrees : treeNumbers) {
            // ランダムフォレストの設定
            RandomForest rf = new RandomForest();
            rf.setNumIterations(numTrees);  // 木の本数を設定
            rf.setSeed(1);  // シード値を固定
            
            // モデルの学習
            rf.buildClassifier(data);

            // 10分割交差検証の実行
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(rf, data, 10, new Random(1));

            // 結果の表示
            double accuracy = eval.pctCorrect();
            int correct = (int)eval.correct();
            int total = (int)(eval.correct() + eval.incorrect());
            
            System.out.printf("%d\t\t%.2f\t\t%d/%d\n", 
                            numTrees, accuracy, correct, total);
        }
        
        System.out.println("\n※ Excelでグラフ化してください");
        System.out.println("X軸: 木の本数 (1, 5, 10, 50, 100, 500)");
        System.out.println("Y軸: 精度 (%)");
    }
}