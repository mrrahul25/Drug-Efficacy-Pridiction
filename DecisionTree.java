import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class DecisionTree {

    public static void main(String[] args) {
        ArrayList<String[]> data = loadData("data.csv");
        ArrayList<String> actualLabels = new ArrayList<>();
        ArrayList<String[]> trainData = new ArrayList<>();
        ArrayList<String[]> testData = new ArrayList<>();

        // Split the data into training and testing sets
        int splitIndex = (int) (0.8 * data.size());
        for (int i = 0; i < data.size(); i++) {
            if (i < splitIndex) {
                trainData.add(data.get(i));
            } else {
                testData.add(data.get(i));
                actualLabels.add(data.get(i)[0]);
            }
        }

        // Train the decision tree
        Node tree = decisionTreeLearning(trainData);

        // Make predictions on the test data
        ArrayList<String> predictions = new ArrayList<>();
        for (String[] sample : testData) {
            predictions.add(predict(tree, sample));
        }

        // Evaluate the predictions
        double[] evaluation = evaluate(predictions, actualLabels);
        double accuracy = evaluation[0];
        double precision = evaluation[1];

        System.out.println("Accuracy: " + accuracy);
        System.out.println("Precision: " + precision);
    }

    public static ArrayList<String[]> loadData(String filename) {
        ArrayList<String[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            br.readLine(); // Skip the header row
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                data.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public static double calculateEntropy(ArrayList<String[]> data) {
        int totalCount = data.size();
        Map<String, Integer> classCounts = new HashMap<>();
        for (String[] row : data) {
            String label = row[0];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        double entropy = 0;
        for (String label : classCounts.keySet()) {
            double probability = (double) classCounts.get(label) / totalCount;
            entropy -= probability * (Math.log(probability) / Math.log(2));
        }
        return entropy;
    }

    public static ArrayList<String[]> splitData(ArrayList<String[]> data, int attributeIndex, String value) {
        ArrayList<String[]> trueRows = new ArrayList<>();
        ArrayList<String[]> falseRows = new ArrayList<>();
        for (String[] row : data) {
            if (row[attributeIndex].equals(value)) {
                trueRows.add(row);
            } else {
                falseRows.add(row);
            }
        }
        return trueRows;
    }

    public static double calculateInformationGain(ArrayList<String[]> data, int attributeIndex) {
        double totalEntropy = calculateEntropy(data);
        Map<String, Integer> valueCounts = new HashMap<>();
        for (String[] row : data) {
            valueCounts.put(row[attributeIndex], valueCounts.getOrDefault(row[attributeIndex], 0) + 1);
        }
        double newEntropy = 0;
        for (String value : valueCounts.keySet()) {
            ArrayList<String[]> subset = splitData(data, attributeIndex, value);
            double probability = (double) valueCounts.get(value) / data.size();
            newEntropy += probability * calculateEntropy(subset);
        }
        return totalEntropy - newEntropy;
    }

    public static int findBestSplit(ArrayList<String[]> data) {
        double bestInformationGain = 0;
        int bestAttribute = -1;
        for (int i = 1; i < data.get(0).length; i++) {
            double informationGain = calculateInformationGain(data, i);
            if (informationGain > bestInformationGain) {
                bestInformationGain = informationGain;
                bestAttribute = i;
            }
        }
        return bestAttribute;
    }

    public static Node decisionTreeLearning(ArrayList<String[]> data) {
        if (data.stream().map(row -> row[0]).distinct().count() == 1) {
            return new Node(data.get(0)[0]);
        }
        if (data.get(0).length == 1) {
            Map<String, Integer> classCounts = new HashMap<>();
            for (String[] row : data) {
                classCounts.put(row[0], classCounts.getOrDefault(row[0], 0) + 1);
            }
            String majorityClass = classCounts.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(null);
            return new Node(majorityClass);
        }
        int bestAttribute = findBestSplit(data);
        ArrayList<String[]> trueRows = splitData(data, bestAttribute, data.get(0)[bestAttribute]);
        ArrayList<String[]> falseRows = splitData(data, bestAttribute, data.get(0)[bestAttribute]);
        Node trueBranch = decisionTreeLearning(trueRows);
        Node falseBranch = decisionTreeLearning(falseRows);
        return new Node(bestAttribute, trueBranch, falseBranch);
    }

    public static String predict(Node tree, String[] sample) {
        if (tree.isLeaf()) {
            return tree.getLabel();
        }
        int attribute = tree.getAttribute();
        if (sample[attribute].equals(sample[attribute])) {
            return predict(tree.getTrueBranch(), sample);
        } else {
            return predict(tree.getFalseBranch(), sample);
        }
    }

    public static double[] evaluate(ArrayList<String> predictions, ArrayList<String> actual) {
        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;
        for (int i = 0; i < predictions.size(); i++) {
            if (predictions.get(i).equals("1") && actual.get(i).equals("1")) {
                truePositives++;
            } else if (predictions.get(i).equals("1") && actual.get(i).equals("0")) {
                falsePositives++;
            } else if (predictions.get(i).equals("0") && actual.get(i).equals("0")) {
                trueNegatives++;
            } else if (predictions.get(i).equals("0") && actual.get(i).equals("1")) {
                falseNegatives++;
            }
        }
        double accuracy = (double) (truePositives + trueNegatives) / predictions.size();
        double precision = (double) truePositives / (truePositives + falsePositives);
        return new double[]{accuracy, precision};
    }

    static class Node {
        private String label;
        private int attribute;
        private Node trueBranch;
        private Node falseBranch;

        public Node(String label) {
            this.label = label;
        }

        public Node(int attribute, Node trueBranch, Node falseBranch) {
            this.attribute = attribute;
            this.trueBranch = trueBranch;
            this.falseBranch = falseBranch;
        }

        public boolean isLeaf() {
            return attribute == 0;
        }

        public String getLabel() {
            return label;
        }

        public int getAttribute() {
            return attribute;
        }

        public Node getTrueBranch() {
            return trueBranch;
        }

        public Node getFalseBranch() {
            return falseBranch;
        }
    }
}
