/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.nn_weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;


public class Nn_weka {

    static class ParameterResult {
        double learningRate;
        double momentum;
        int trainingTime;
        String hiddenLayers;
        double accuracy;
        String confusionMatrix;
        String parameterString;
        
        public ParameterResult(double lr, double mom, int tt, String hl, double acc, String cm) {
            this.learningRate = lr;
            this.momentum = mom;
            this.trainingTime = tt;
            this.hiddenLayers = hl;
            this.accuracy = acc;
            this.confusionMatrix = cm;
            this.parameterString = String.format("LR=%.3f, M=%.3f, T=%d, H=%s", lr, mom, tt, hl);
        }
    }

    public static void main(String[] args) throws Exception {
        // ARFFファイルの読み込み
        DataSource source = new DataSource("breast-cancer.arff");
        Instances data = source.getDataSet();

        // クラスラベルのインデックスを設定（最後の属性をクラスラベルとして設定）
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("=== ニューラルネットワーク分類実験 (breast-cancer.arff) ===");
        System.out.println("データセット情報:");
        System.out.println("- インスタンス数: " + data.numInstances());
        System.out.println("- 属性数: " + data.numAttributes());
        System.out.println("- クラス数: " + data.numClasses());
        System.out.println();

        // パラメータの組み合わせを定義
        double[] learningRates = {0.1, 0.3, 0.5, 0.7};
        double[] momentums = {0.2, 0.4, 0.6, 0.8};
        int[] trainingTimes = {50, 100, 200, 500};
        String[] hiddenLayerConfigs = {"a", "5", "10", "15", "5,3", "10,5"};

        List<ParameterResult> results = new ArrayList<>();

        // 全パラメータ組み合わせを試行
        int totalCombinations = learningRates.length * momentums.length * trainingTimes.length * hiddenLayerConfigs.length;
        int currentCombination = 0;

        for (double lr : learningRates) {
            for (double mom : momentums) {
                for (int tt : trainingTimes) {
                    for (String hl : hiddenLayerConfigs) {
                        currentCombination++;
                        System.out.printf("実験 %d/%d: LR=%.1f, M=%.1f, T=%d, H=%s\n", 
                                        currentCombination, totalCombinations, lr, mom, tt, hl);

                        try {
                            // ニューラルネットワークの設定
                            MultilayerPerceptron mlp = new MultilayerPerceptron();
                            mlp.setLearningRate(lr);
                            mlp.setMomentum(mom);
                            mlp.setTrainingTime(tt);
                            mlp.setHiddenLayers(hl);

                            // モデルの学習
                            mlp.buildClassifier(data);

                            // 10分割交差検証の実行
                            Evaluation eval = new Evaluation(data);
                            eval.crossValidateModel(mlp, data, 10, new Random(1));

                            // 結果を保存
                            double accuracy = eval.pctCorrect();
                            String confusionMatrix = eval.toMatrixString();
                            results.add(new ParameterResult(lr, mom, tt, hl, accuracy, confusionMatrix));

                            System.out.printf("  -> 精度: %.2f%%\n", accuracy);

                        } catch (Exception e) {
                            System.out.println("  -> エラー: " + e.getMessage());
                        }
                    }
                }
            }
        }

        // 結果をソートして上位5件を表示
        results.sort((a, b) -> Double.compare(b.accuracy, a.accuracy));

        System.out.println("\n=== 実験結果 (上位5件) ===");
        for (int i = 0; i < Math.min(5, results.size()); i++) {
            ParameterResult result = results.get(i);
            System.out.printf("\n第%d位: 精度 %.2f%%\n", i + 1, result.accuracy);
            System.out.println("パラメータ: " + result.parameterString);
            System.out.println("混同行列:");
            System.out.println(result.confusionMatrix);
        }

        // 全結果の統計情報
        System.out.println("\n=== 統計情報 ===");
        double avgAccuracy = results.stream().mapToDouble(r -> r.accuracy).average().orElse(0.0);
        double maxAccuracy = results.stream().mapToDouble(r -> r.accuracy).max().orElse(0.0);
        double minAccuracy = results.stream().mapToDouble(r -> r.accuracy).min().orElse(0.0);
        
        System.out.printf("平均精度: %.2f%%\n", avgAccuracy);
        System.out.printf("最高精度: %.2f%%\n", maxAccuracy);
        System.out.printf("最低精度: %.2f%%\n", minAccuracy);
        System.out.printf("実験総数: %d\n", results.size());
    }
}