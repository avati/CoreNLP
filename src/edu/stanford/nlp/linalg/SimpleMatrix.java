package edu.stanford.nlp.linalg;

import java.util.Random;
import java.io.PrintStream;
import java.util.PriorityQueue;

/**
 * Contains a class that mimics org.ejml.simple.SimpleMatrix
 * and is a wrapper around org.ejml.simple.SimpleMatrix.
 *
 * We use this "outer" class to implement our bulk/async matrix
 * multiplication operators.
 *
 * @author Anand Avati
 */
public class SimpleMatrix implements java.io.Serializable {
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

    public SimpleMatrix extractVector(boolean extractRow, int elem) {
	return new SimpleMatrix(inner.extractVector(extractRow, elem));
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

    /* This is the asynchronous matrix multiplier. It
       accepts the matrix B, and returns immediately.

       Sometime eventually, the matrix B is actually
       multiplied, and the result is notified
       through the callback.
    */

    public void mult_async(SimpleMatrix b, SimpleMatrixNotifier n) {
	mult_async(b, false, false, null, n);
    }

    public void mult_async(SimpleMatrix b, SimpleMatrix addToResult, SimpleMatrixNotifier n) {
	mult_async(b, false, false, addToResult, n);
    }

    class Multiplier {
	public SimpleMatrix right;
	public SimpleMatrix addToResult;
	public SimpleMatrixNotifier n;
	public int depth;
	public Multiplier(SimpleMatrix right, SimpleMatrix addToResult, SimpleMatrixNotifier n, int depth) {
	    this.right = right;
	    this.addToResult = addToResult;
	    this.n = n;
	    this.depth = depth;
	}
	public Multiplier(SimpleMatrix right, SimpleMatrix addToResult, SimpleMatrixNotifier n) {
	    this(right, addToResult, n, 0);
	}
    }

    transient PriorityQueue<Multiplier> queue;
    transient PriorityQueue<Multiplier> transposeQueue;

    public void mult_async(SimpleMatrix b, boolean leftTranspose, boolean rightTranspose, SimpleMatrix addToResult, SimpleMatrixNotifier n) {
	/* Batching to be done later */
	if (queue == null)
	    queue = new PriorityQueue<Multiplier>((Multiplier left, Multiplier right) -> left.depth - right.depth);
	if (transposeQueue == null)
	    transposeQueue = new PriorityQueue<Multiplier>((Multiplier left, Multiplier right) -> left.depth - right.depth);

	if (!leftTranspose)
	    queue.add(new Multiplier(rightTranspose ? b.transpose() : b, addToResult, n));
	else
	    transposeQueue.add(new Multiplier(rightTranspose ? b.transpose() : b, addToResult, n));
    }

    public int churn(PriorityQueue<Multiplier> queue, SimpleMatrix left) {
	if (queue.size() == 0)
	    return 0;

	Multiplier[] requests = queue.toArray(new Multiplier[0]);
	queue.clear();

	int ncols = 0;
	int nrows = requests[0].right.numRows();
	for (Multiplier request : requests)
	    ncols += request.right.numCols();
	
	SimpleMatrix right = new SimpleMatrix(nrows, ncols);
	int c = 0;
	for (Multiplier request: requests) {
	    right.insertIntoThis(0, c, request.right);
	    c += request.right.numCols();
	}

	SimpleMatrix result = left.mult(right);

	c = 0;
	for (Multiplier request: requests) {
	    SimpleMatrix ans = result.extractMatrix(0, result.numRows(), c, c + request.right.numCols());
	    c += request.right.numCols();
	    request.n.notify(request.addToResult == null ? ans : ans.plus(request.addToResult));
	}

	return requests.length;
    }

    public int churn() {
	int cnt = 0;
	if (queue != null) {
	    int n = churn(queue, this);
	    //	    System.out.printf("Mult batch: %d\n", n);
	    cnt += n;
	}
	if (transposeQueue != null) {
	    int n = churn(transposeQueue, this.transpose());
	    //	    System.out.printf("MultT batch: %d\n", n);
	    cnt += n;
	}
	return cnt;
    }
}

