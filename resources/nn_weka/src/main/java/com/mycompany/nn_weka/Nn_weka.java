/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.nn_weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import java.util.Random;


public class Nn_weka {

    public static void main(String[] args) throws Exception {
        // ARFFファイルの読み込み
        DataSource source = new DataSource("iris.arff");
        Instances data = source.getDataSet();

        // クラスラベルのインデックスを設定（最後の属性をクラスラベルとして設定）
        data.setClassIndex(data.numAttributes() - 1);

        // ニューラルネットワークの設定
        MultilayerPerceptron mlp = new MultilayerPerceptron();   
        mlp.setLearningRate(0.1);
        mlp.setMomentum(0.2);
        mlp.setTrainingTime(10);    // 非常に少ない回数に設定してある．この状態で精度を確認してみよう．
        mlp.setHiddenLayers("a");        
        // aを設定すると自動で中間層を設定してくれる
        // aの場合，入力データの特徴量とクラス数を足して2で割ったものを中間層の数とする，
        // 1層のネットワークを構築．
        // mlp.setHiddenLayers("5, 10");   
        // 自身で設定したい場合はこのようにして各層のノード数を記載する．

        // モデルの学習
        mlp.buildClassifier(data);

        // 10分割交差検証の実行
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(mlp, data, 10, new Random(1));

        // 結果の表示
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        // 隠れ層の詳細を表示
        printHiddenLayerDetails(mlp, data);
    }

    private static void printHiddenLayerDetails(MultilayerPerceptron mlp, Instances data) {
        // getHiddenLayersメソッドを使用して隠れ層の設定を取得
        String hiddenLayers = mlp.getHiddenLayers();
        System.out.println("\nHidden Layers:");
        System.out.println("Hidden layers specification: " + hiddenLayers);

        if (hiddenLayers.equals("a")) {
            int numAttributes = data.numAttributes() - 1; // クラス属性を除く
            int numClasses = data.numClasses();
            int numNodes = (numAttributes + numClasses) / 2;
            System.out.println("The network has one hidden layer with " + numNodes + " nodes.");
        } else {
            String[] layers = hiddenLayers.split(",");
            System.out.println("The network has " + layers.length + " hidden layers.");
            for (int i = 0; i < layers.length; i++) {
                System.out.println("Layer " + (i + 1) + " has " + layers[i] + " nodes.");
            }
        }
    }
}