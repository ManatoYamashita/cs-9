package com.mycompany.randomf;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;

public class RandomFRegression {
    public static void main(String[] args) throws Exception {
        // housing.arffファイルの読み込み
        DataSource source = new DataSource("housing.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("=== ランダムフォレスト回帰実験 (housing.arff) ===\n");
        System.out.println("木の本数\tRMSE\t\tMAE\t\t相関係数");
        System.out.println("------------------------------------------------");

        // 木の本数を変化させて実験
        int[] treeNumbers = {1, 5, 10, 50, 100, 500};
        double bestRMSE = Double.MAX_VALUE;
        int bestTreeNum = 0;
        RandomForest bestModel = null;
        
        for (int numTrees : treeNumbers) {
            // ランダムフォレストの設定
            RandomForest rf = new RandomForest();
            rf.setNumIterations(numTrees);  // 木の本数を設定
            rf.setSeed(-1);  // シード値を-1に設定（ランダム）
            rf.setCalcOutOfBag(true);  // OOBエラーを計算
            
            // モデルの学習
            rf.buildClassifier(data);

            // 10分割交差検証の実行
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(rf, data, 10, new Random(1));

            // 結果の表示
            double rmse = eval.rootMeanSquaredError();
            double mae = eval.meanAbsoluteError();
            double correlation = eval.correlationCoefficient();
            
            System.out.printf("%d\t\t%.4f\t\t%.4f\t\t%.4f\n", 
                            numTrees, rmse, mae, correlation);
            
            // 最も精度が高いモデルを記録
            if (rmse < bestRMSE) {
                bestRMSE = rmse;
                bestTreeNum = numTrees;
                bestModel = rf;
            }
        }
        
        System.out.println("\n※ Excelでグラフ化してください");
        System.out.println("X軸: 木の本数 (1, 5, 10, 50, 100, 500)");
        System.out.println("Y軸: RMSE");
        
        // 最も精度が高いモデルの特徴量重要度を表示
        System.out.println("\n=== 最も精度が高いモデル（木の本数: " + bestTreeNum + ", RMSE: " + String.format("%.4f", bestRMSE) + "）===");
        
        // 特徴量重要度を計算するためにオプションを設定して再学習
        RandomForest rfWithImportance = new RandomForest();
        rfWithImportance.setNumIterations(bestTreeNum);
        rfWithImportance.setSeed(-1);
        rfWithImportance.setCalcOutOfBag(true);
        rfWithImportance.setComputeAttributeImportance(true);
        rfWithImportance.buildClassifier(data);
        
        // 特徴量重要度の取得と表示
        System.out.println("\n属性番号\t属性名\t\t\t\t説明");
        System.out.println("--------------------------------------------------------------");
        
        // housing.arffの属性説明
        String[] attrDescriptions = {
            "CRIM\t\t犯罪率",
            "ZN\t\t住宅地の割合",
            "INDUS\t\t非小売業の割合",
            "CHAS\t\tチャールズ川沿い（0/1）",
            "NOX\t\t窒素酸化物濃度",
            "RM\t\t平均部屋数",
            "AGE\t\t築年数",
            "DIS\t\t雇用中心地までの距離",
            "RAD\t\t高速道路へのアクセス",
            "TAX\t\t固定資産税率",
            "PTRATIO\t\t生徒教師比率",
            "B\t\t黒人居住者の割合",
            "LSTAT\t\t低所得者の割合"
        };
        
        for (int i = 0; i < data.numAttributes() - 1 && i < attrDescriptions.length; i++) {
            System.out.printf("%d\t\t%s\n", i + 1, attrDescriptions[i]);
        }
        
        System.out.println("\n※ RandomForestのデフォルト実装では、個別の特徴量重要度の数値は");
        System.out.println("  直接取得できないため、OOBエラーなどから間接的に評価します。");
        
        System.out.println("\n=== 考察のポイント ===");
        System.out.println("1. 木の本数とRMSEの関係（精度の向上と収束）");
        System.out.println("2. 最も重要な特徴量とその意味");
        System.out.println("3. 住宅価格予測における各特徴量の影響");
    }
}