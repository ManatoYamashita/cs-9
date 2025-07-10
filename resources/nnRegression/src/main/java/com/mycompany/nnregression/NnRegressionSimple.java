package com.mycompany.nnregression;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import java.util.Random;

public class NnRegressionSimple {
    public static void main(String[] args) throws Exception {
        // ARFFファイルの読み込み
        DataSource source = new DataSource("housing.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("=== ニューラルネットワーク回帰実験 (housing.arff) ===\n");

        // 5つの異なるパラメータ設定を試す
        double[] learningRates = {0.1, 0.05, 0.2, 0.15, 0.3};
        double[] momentums = {0.2, 0.5, 0.3, 0.6, 0.4};
        int[] trainingTimes = {200, 300, 400, 250, 350};
        String[] hiddenLayers = {"a", "10", "20", "15,10", "10,5"};

        for (int i = 0; i < 5; i++) {
            System.out.printf("実験 %d: LR=%.2f, M=%.1f, T=%d, H=%s\n", 
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
            System.out.println("回帰性能指標:");
            System.out.printf("  RMSE: %.4f\n", eval.rootMeanSquaredError());
            System.out.printf("  MAE:  %.4f\n", eval.meanAbsoluteError());
            System.out.printf("  相関係数: %.4f\n", eval.correlationCoefficient());
            System.out.println("----------------------------------------\n");
        }
    }
}