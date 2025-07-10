package com.mycompany.nn_weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import java.util.Random;

public class NnWekaSimple {
    public static void main(String[] args) throws Exception {
        // ARFFファイルの読み込み
        DataSource source = new DataSource("breast-cancer.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("=== ニューラルネットワーク分類実験 (breast-cancer.arff) ===\n");

        // 5つの異なるパラメータ設定を試す
        double[] learningRates = {0.1, 0.3, 0.5, 0.2, 0.4};
        double[] momentums = {0.2, 0.5, 0.8, 0.6, 0.3};
        int[] trainingTimes = {100, 200, 300, 150, 250};
        String[] hiddenLayers = {"a", "10", "20", "10,5", "15,10"};

        for (int i = 0; i < 5; i++) {
            System.out.printf("実験 %d: LR=%.1f, M=%.1f, T=%d, H=%s\n", 
                            i + 1, learningRates[i], momentums[i], trainingTimes[i], hiddenLayers[i]);

            // ニューラルネットワークの設定
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setLearningRate(learningRates[i]);
            mlp.setMomentum(momentums[i]);
            mlp.setTrainingTime(trainingTimes[i]);
            mlp.setHiddenLayers(hiddenLayers[i]);

            // モデルの学習
            mlp.buildClassifier(data);

            // 10分割交差検証の実行
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(mlp, data, 10, new Random(1));

            // 結果の表示
            System.out.printf("精度: %.2f%% (%d/%d)\n", 
                            eval.pctCorrect(), 
                            (int)eval.correct(), 
                            (int)(eval.correct() + eval.incorrect()));
            System.out.println("混同行列:");
            System.out.println(eval.toMatrixString());
            System.out.println("----------------------------------------\n");
        }
    }
}