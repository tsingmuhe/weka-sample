package com.sunchp.weka.sample;

import com.google.common.io.Resources;
import org.junit.Test;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class AppTest {

    private Instances loadData(String name) throws IOException {
        Instances data = new Instances(new BufferedReader(new FileReader(Resources.getResource(name).getFile())));
        data.setClassIndex(4);
        return data;
    }

    @Test
    public void testInstances() throws IOException {
        Instances data = loadData("data/weather.nominal.arff");
        System.out.println(data.numInstances());
        System.out.println(data);

    }

    @Test
    public void testRemoveFilter() throws Exception {
        Instances data = loadData("data/weather.nominal.arff");

        Remove remove = new Remove();
        String[] opts = new String[]{"-R", "5"};
        remove.setOptions(opts);
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);

        System.out.println(data.numInstances());
        System.out.println(data);
    }

    @Test
    public void testSelect() throws Exception {
        Instances data = loadData("data/weather.nominal.arff");

        AttributeSelection attSelection = new AttributeSelection();
        attSelection.setEvaluator(new InfoGainAttributeEval());
        attSelection.setSearch(new Ranker());
        attSelection.SelectAttributes(data);

        int num = attSelection.numberAttributesSelected();
        int[] indices = attSelection.selectedAttributes();

        System.out.println(num);
        System.out.println(Utils.arrayToString(indices));
    }

    @Test
    public void testTrain() throws Exception {
        Instances data = loadData("data/weather.nominal.arff");

        AttributeSelection attSelection = new AttributeSelection();
        attSelection.setEvaluator(new InfoGainAttributeEval());
        attSelection.setSearch(new Ranker());
        attSelection.SelectAttributes(data);

        J48 tree = new J48();
        String[] optsJ48 = new String[]{"-U"};
        tree.setOptions(optsJ48);
        tree.buildClassifier(data);

        System.out.println(tree);
    }

    @Test
    public void testTreeVisualizer() throws Exception {
        Instances data = loadData("data/weather.nominal.arff");

        AttributeSelection attSelection = new AttributeSelection();
        attSelection.setEvaluator(new InfoGainAttributeEval());
        attSelection.setSearch(new Ranker());
        attSelection.SelectAttributes(data);

        J48 tree = new J48();
        String[] optsJ48 = new String[]{"-U"};
        tree.setOptions(optsJ48);
        tree.buildClassifier(data);

        TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
        JFrame frame = new JFrame("Tree Visualizer");
        frame.setSize(800, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(tv);
        frame.setVisible(true);
        tv.fitToScreen();
    }


    @Test
    public void testClassifyInstance() throws Exception {
        Instances dataTrain = loadData("data/weather.nominal.arff");
        Instances dataTest = loadData("data/weather.nominal.arff");

        AttributeSelection attSelection = new AttributeSelection();
        attSelection.setEvaluator(new InfoGainAttributeEval());
        attSelection.setSearch(new Ranker());
        attSelection.SelectAttributes(dataTrain);

        J48 tree = new J48();
        String[] optsJ48 = new String[]{"-U"};
        tree.setOptions(optsJ48);
        tree.buildClassifier(dataTrain);

        double right = 0;
        int sum = dataTest.numInstances();

        dataTest.setClassIndex(4);
        for (int i = 0; i < sum; i++) {
            double result = tree.classifyInstance(dataTest.instance(i));
            System.out.println(result);
            if (result == dataTest.instance(i).classValue()) {
                right++;
            }
        }

        System.out.println("J48 classification precision: " + (right / sum) * 100 + "%");
    }
}
