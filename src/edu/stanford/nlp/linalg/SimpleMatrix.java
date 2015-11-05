package edu.stanford.nlp.linalg;

import java.util.Random;
import java.io.PrintStream;

/**
 * Contains a class that mimics org.ejml.simple.SimpleMatrix
 * and is a wrapper around org.ejml.simple.SimpleMatrix.
 *
 * We use this "outer" class to implement our bulk/async matrix
 * multiplication operators.
 *
 * @author Anand Avati
 */
public class SimpleMatrix {
    org.ejml.simple.SimpleMatrix inner;

    public SimpleMatrix(int nrows, int ncols) {
	inner = new org.ejml.simple.SimpleMatrix(nrows, ncols);
    }

    public SimpleMatrix(SimpleMatrix orig) {
	inner = new org.ejml.simple.SimpleMatrix(orig.inner);
    }

    public SimpleMatrix(double[][] data) {
	inner = new org.ejml.simple.SimpleMatrix(data);
    }

    public SimpleMatrix(org.ejml.simple.SimpleMatrix innerMatrix) {
	inner = innerMatrix;
    }

    public void insertIntoThis(int insertRow, int insertCol, SimpleMatrix b) {
	inner.insertIntoThis(insertRow, insertCol, b.inner);
    }

    public SimpleMatrix plus(double beta, SimpleMatrix b) {
	return new SimpleMatrix(inner.plus(beta, b.inner));
    }

    public SimpleMatrix plus(SimpleMatrix b) {
	return plus(0.0, b);
    }

    public SimpleMatrix divide(double val) {
	return new SimpleMatrix(inner.divide(val));
    }

    public int getNumElements() {
	return inner.getNumElements();
    }

    public SimpleMatrix scale(double val) {
	return new SimpleMatrix(inner.scale(val));
    }

    public static SimpleMatrix random(int numRows, int numCols, double minValue, double maxValue, Random rand) {
	return new SimpleMatrix(org.ejml.simple.SimpleMatrix.random(numRows, numCols, minValue, maxValue, rand));
    }

    public int numRows() {
	return inner.numRows();
    }

    public int numCols() {
	return inner.numCols();
    }

    public static SimpleMatrix loadBinary(String fileName) throws java.io.IOException {
	return new SimpleMatrix(org.ejml.simple.SimpleMatrix.loadBinary(fileName));
    }

    public static SimpleMatrix identity(int width) {
	return new SimpleMatrix(org.ejml.simple.SimpleMatrix.identity(width));
    }

    public double get(int index) {
	return inner.get(index);
    }

    public double get(int row, int col) {
	return inner.get(row, col);
    }

    public void set(double val) {
	inner.set(val);
    }

    public void set(int index, double val) {
	inner.set(index, val);
    }

    public void set (int row, int col, double val) {
	inner.set(row, col, val);
    }

    public SimpleMatrix mult(SimpleMatrix b) {
	return new SimpleMatrix(inner.mult(b.inner));
    }

    public SimpleMatrix extractMatrix(int y0, int y1, int x0, int x1) {
	return new SimpleMatrix(inner.extractMatrix(y0, y1, x0, x1));
    }

    public SimpleMatrix transpose() {
	return new SimpleMatrix(inner.transpose());
    }

    public SimpleMatrix elementMult(SimpleMatrix b) {
	return new SimpleMatrix(inner.elementMult(b.inner));
    }

    public double elementSum() {
	return inner.elementSum();
    }

    public double dot(SimpleMatrix b) {
	return inner.dot(b.inner);
    }

    public SimpleMatrix minus(SimpleMatrix b) {
	return new SimpleMatrix(inner.minus(b.inner));
    }

    public double normF() {
	return inner.normF();
    }

    public org.ejml.data.DenseMatrix64F getMatrix() {
	return inner.getMatrix();
    }
}

