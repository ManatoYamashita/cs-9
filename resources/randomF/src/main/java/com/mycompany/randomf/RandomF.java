/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.randomf;


import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;

public class RandomF {

    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("iris.arff");
        Instances data = source.getDataSet();

        // クラスラベルのインデックスを設定（最後の属性をクラスラベルとして設定）
        data.setClassIndex(data.numAttributes() - 1);

        // ランダムフォレストの設定
        RandomForest rf = new RandomForest();
        String[] options = {
            "-I", "100",    // ツリーの数
            "-K", "0",      // 各木でランダムに選択する属性の数（0は√(numAttributes)）
            "-S", "1",       // シード値の設定
            "-attribute-importance",    // 特徴量の重要度を表示
            "-print"        // 構築した各木の構造を出力
        };
        rf.setOptions(options);

        // モデルの学習
        rf.buildClassifier(data);

        // 10分割交差検証の実行
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(rf, data, 10, new Random(1));

        // 結果の表示
        System.out.println(rf);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        // 以下，二つの出力は回帰の時にはコメントアウトすること．（エラーになります）
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());        
    }
}
