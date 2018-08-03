package com.sunchp.weka.sample;

import com.google.common.io.Resources;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;

public class App {
    Instances m_Data = null;
    Classifier m_Classifier = null;

    public App() throws Exception {
        m_Classifier = (J48) SerializationHelper.read(new FileInputStream("oos.model"));
        m_Data = loadData("data/weather.nominal.arff");
    }

    private Instances loadData(String name) throws IOException {
        Instances data = new Instances(new BufferedReader(new FileReader(Resources.getResource(name).getFile())));
        data.setClassIndex(4);
        return data;
    }

    public void classifyMessage(String _outlook, String _temperature, String _humidity, String _windy) throws Exception {
        Instances testset = m_Data.stringFreeStructure();
        Instance instance = makeInstance(_outlook, _temperature, _humidity, _windy, testset);
        System.out.println(m_Data.numAttributes());
        System.out.println(instance);
        double predicted = m_Classifier.classifyInstance(instance);
        System.out.println("predicted:" + predicted);
        System.out.println(m_Data.classAttribute());
        System.out.println("Message classified as : " + m_Data.classAttribute().value((int) predicted));
    }

    private Instance makeInstance(String _outlook, String _temperature, String _humidity, String _windy, Instances data) {
        Instance instance = new DenseInstance(4);
        instance.setDataset(data);

        Attribute outlook = data.attribute("outlook");
        Attribute temperature = data.attribute("temperature");
        Attribute humidity = data.attribute("humidity");
        Attribute windy = data.attribute("windy");

        instance.setValue(outlook, _outlook);
        instance.setValue(temperature, _temperature);
        instance.setValue(humidity, _humidity);
        instance.setValue(windy, _windy);

        return instance;
    }

    public static void main(String[] args) throws Exception {
        App wTestInstance = new App();
        wTestInstance.classifyMessage("sunny", "hot", "high", "FALSE");
        wTestInstance.classifyMessage("overcast", "hot", "high", "FALSE");
    }
}
