/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.nnregression;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;

import java.util.Random;

public class NnRegression {

    public static void main(String[] args) throws Exception {
        // ARFFファイルの読み込み
        DataSource source = new DataSource("housing.arff");
        Instances data = source.getDataSet();

        // クラスラベルのインデックスを設定（最後の属性を回帰ターゲットとして設定）
        data.setClassIndex(data.numAttributes() - 1);

        // ニューラルネットワークの設定
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        
        // オプションの設定
        String[] options = {
            "-L", "0.1",    // 学習率
            "-M", "0",      // 慣性項
            "-N", "10",    // エポック数
            "-H", "a"       // 隠れ層の構造
            //"-H", "5, 10" // このようにして，陽に与えても構わないです    
        };
        mlp.setOptions(options);
        //その他のオプションは
        // https://weka.sourceforge.io/doc.dev/weka/classifiers/functions/MultilayerPerceptron.html#setOptions-java.lang.String:A-
        // を参照してみてください．
               
        // モデルの学習
        mlp.buildClassifier(data);


        // 10分割交差検証の実行
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(mlp, data, 10, new Random(1));

        // 結果の表示
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
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
