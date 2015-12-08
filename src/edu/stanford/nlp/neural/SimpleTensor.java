package edu.stanford.nlp.neural;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.PriorityQueue;
import java.util.NoSuchElementException;

import edu.stanford.nlp.linalg.SimpleMatrix;
import edu.stanford.nlp.linalg.SimpleMatrixNotifier;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This class defines a block tensor, somewhat like a three
 * dimensional matrix.  This can be created in various ways, such as
 * by providing an array of SimpleMatrix slices, by providing the
 * initial size to create a 0-initialized tensor, or by creating a
 * random matrix.
 *
 * @author John Bauer
 * @author Richard Socher
 */
public class SimpleTensor implements Serializable {
  private final SimpleMatrix[] slices;

  final int numRows;
  final int numCols;
  final int numSlices;

  /**
   * Creates a zero initialized tensor
   */
  public SimpleTensor(int numRows, int numCols, int numSlices) {
    slices = new SimpleMatrix[numSlices];
    for (int i = 0; i < numSlices; ++i) {
      slices[i] = new SimpleMatrix(numRows, numCols);
    }

    this.numRows = numRows;
    this.numCols = numCols;
    this.numSlices = numSlices;
  }

  /**
   * Copies the data in the slices.  Slices are copied rather than
   * reusing the original SimpleMatrix objects.  Each slice must be
   * the same size.
   */
  public SimpleTensor(SimpleMatrix[] slices) {
    this.numRows = slices[0].numRows();
    this.numCols = slices[0].numCols();
    this.numSlices = slices.length;
    this.slices = new SimpleMatrix[slices.length];
    for (int i = 0; i < numSlices; ++i) {
      if (slices[i].numRows() != numRows || slices[i].numCols() != numCols) {
        throw new IllegalArgumentException("Slice " + i + " has matrix dimensions " + slices[i].numRows() + "," + slices[i].numCols() + ", expected " + numRows + "," + numCols);
      }
      this.slices[i] = new SimpleMatrix(slices[i]);
    }
    
  }

  /**
   * Returns a randomly initialized tensor with values draft from the
   * uniform distribution between minValue and maxValue.
   */
  public static SimpleTensor random(int numRows, int numCols, int numSlices, double minValue, double maxValue, java.util.Random rand) {
    SimpleTensor tensor = new SimpleTensor(numRows, numCols, numSlices);
    for (int i = 0; i < numSlices; ++i) {
      tensor.slices[i] = SimpleMatrix.random(numRows, numCols, minValue, maxValue, rand);
    }
    return tensor;
  }

  /**
   * Number of rows in the tensor
   */
  public int numRows() {
    return numRows;
  }

  /**
   * Number of columns in the tensor
   */
  public int numCols() {
    return numCols;
  }

  /**
   * Number of slices in the tensor
   */
  public int numSlices() {
    return numSlices;
  }

  /**
   * Total number of elements in the tensor
   */
  public int getNumElements() {
    return numRows * numCols * numSlices;
  }

  public void set(double value) {
    for (int slice = 0; slice < numSlices; ++slice) {
      slices[slice].set(value);
    }
  }

  /**
   * Returns a new tensor which has the values of the original tensor
   * scaled by <code>scaling</code>.  The original object is
   * unaffected.
   */
  public SimpleTensor scale(double scaling) {
    SimpleTensor result = new SimpleTensor(numRows, numCols, numSlices);
    for (int slice = 0; slice < numSlices; ++slice) {
      result.slices[slice] = slices[slice].scale(scaling);
    }
    return result;
  }

  /**
   * Returns <code>other</code> added to the current object, which is unaffected.
   */
  public SimpleTensor plus(SimpleTensor other) {
    if (other.numRows != numRows || other.numCols != numCols || other.numSlices != numSlices) {
      throw new IllegalArgumentException("Sizes of tensors do not match.  Our size: " + numRows + "," + numCols + "," + numSlices + "; other size " + other.numRows + "," + other.numCols + "," + other.numSlices);
    }
    SimpleTensor result = new SimpleTensor(numRows, numCols, numSlices);
    for (int i = 0; i < numSlices; ++i) {
      result.slices[i] = slices[i].plus(other.slices[i]);
    }
    return result;
  }

  /**
   * Performs elementwise multiplication on the tensors.  The original
   * objects are unaffected.
   */
  public SimpleTensor elementMult(SimpleTensor other) {
    if (other.numRows != numRows || other.numCols != numCols || other.numSlices != numSlices) {
      throw new IllegalArgumentException("Sizes of tensors do not match.  Our size: " + numRows + "," + numCols + "," + numSlices + "; other size " + other.numRows + "," + other.numCols + "," + other.numSlices);
    }
    SimpleTensor result = new SimpleTensor(numRows, numCols, numSlices);
    for (int i = 0; i < numSlices; ++i) {
      result.slices[i] = slices[i].elementMult(other.slices[i]);
    }
    return result;
  }

  /**
   * Returns the sum of all elements in the tensor.
   */
  public double elementSum() {
    double sum = 0.0;
    for (SimpleMatrix slice : slices) {
      sum += slice.elementSum();
    }
    return sum;
  }

  /**
   * Use the given <code>matrix</code> in place of <code>slice</code>.
   * Does not copy the <code>matrix</code>, but rather uses the actual object.
   */
  public void setSlice(int slice, SimpleMatrix matrix) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    if (matrix.numCols() != numCols) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.numCols() + " columns, tensor has " + numCols);
    }
    if (matrix.numRows() != numRows) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.numRows() + " columns, tensor has " + numRows);
    }
    slices[slice] = matrix;
    leftCache = null;
    leftSCCache = null;
  }

  /**
   * Returns the SimpleMatrix at <code>slice</code>.
   * <br>
   * The actual slice is returned - do not alter this unless you know what you are doing.
   */
  public SimpleMatrix getSlice(int slice) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    return slices[slice];
  }

  /**
   * Returns a column vector where each entry is the nth bilinear
   * product of the nth slices of the two tensors.
   */
  public SimpleMatrix bilinearProducts(SimpleMatrix in) {
    if (in.numCols() != 1) {
      throw new AssertionError("Expected a column vector");
    }
    if (in.numRows() != numCols) {
      throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
    }
    if (numRows != numCols) {
      throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
    }
    SimpleMatrix inT = in.transpose();
    SimpleMatrix out = new SimpleMatrix(numSlices, 1);
    for (int slice = 0; slice < numSlices; ++slice) {
      double result = inT.mult(slices[slice]).mult(in).get(0);
      out.set(slice, result);
    }
    return out;
  }

    class Multiplier {
	public SimpleMatrix right;
	public SimpleMatrixNotifier n;
	public int depth;
	public Multiplier(SimpleMatrix right, SimpleMatrixNotifier n, int depth) {
	    this.right = right;
	    this.n = n;
	    this.depth = depth;
	}
	public Multiplier(SimpleMatrix right, SimpleMatrixNotifier n) {
	    this(right, n, 0);
	}
    }

    transient PriorityQueue<Multiplier> queue;

    public void bilinearProducts_async(SimpleMatrix in, SimpleMatrixNotifier n) {
	if (queue == null)
	    queue = new PriorityQueue<Multiplier>((Multiplier left, Multiplier right) -> left.depth - right.depth);
	if (in.numCols() != 1) {
	    throw new AssertionError("Expected a column vector");
	}
	if (in.numRows() != numCols) {
	    throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
	}
	if (numRows != numCols) {
	    throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
	}

	queue.add(new Multiplier(in, n));
    }

    void print_shape(INDArray arr) {
	int[] shape = arr.shape();

	System.out.printf("Shape: %d", shape[0]);
	for (int i = 1;i < shape.length; i++)
	    System.out.printf("x%d", shape[i]);
	System.out.printf("\n");
	
    }

    transient INDArray leftCache = null;
    transient INDArray ones = null;
    transient int ones_nrows = 0;

    void prepGPUMult() {
	double[][] leftData = new double[numSlices][];
	for (int i = 0; i < numSlices; i++)
	    leftData[i] = slices[i].getMatrix().data;

	leftCache = Nd4j.create(leftData).reshape(numSlices * numRows, numCols); // TODO: fix reshape!! (pre and post transpose)
    }

    void create_ones(int nrows) {
	if (ones_nrows == nrows)
	    return;

	ones = Nd4j.create(new int[]{numSlices, nrows*numSlices});

	for (int d = 0; d < numSlices; d++) {
	    for (int c = 0; c < nrows; c++) {
		ones.put(d, (d*nrows + c), 1.0); // TODO: verify this!!
	    }
	}
	ones_nrows = nrows;
    }
    
    public int churnMultGPU(Multiplier[] requests) {
	int nrows = requests[0].right.numRows();
	create_ones(nrows);
	int ncols = requests.length;

	double[][]rightDataT = new double[requests.length][];
	for (int i = 0; i < requests.length; i++)
	    rightDataT[i] = requests[i].right.getMatrix().data;

	INDArray left = leftCache;
	INDArray rightT = Nd4j.create(rightDataT); //.reshape(new int[]{ncols, nrows});
	INDArray right = rightT.transpose();

	//print_shape(left);
	//print_shape(right);

	INDArray rightB = right.repmat(new int[]{numSlices * nrows, ncols});

	//print_shape(rightB);

	INDArray hadamard_product = left.mmul(right).muli(rightB);
	//print_shape(hadamard_product);
	//print_shape(ones);
	INDArray blp = ones.mmul(hadamard_product);

	//print_shape(blp);
	
	SimpleMatrix result = new SimpleMatrix(numSlices, ncols, true, blp.data().asDouble());

	int c = 0;
	for (Multiplier request: requests) {
	    request.n.notify(result.extractVector(false, c));
	}

	//	System.out.printf("Tensor batch: %d\n", requests.length);
	return requests.length;
    }

    public int churnMult() {
	if (queue == null || queue.size() == 0)
	    return 0;

	Multiplier[] requests = queue.toArray(new Multiplier[0]);
	queue.clear();

	if (leftCache != null) {
	    return churnMultGPU(requests);
	}

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

	SimpleMatrix left = new SimpleMatrix(numRows * numSlices, numCols);
	for (int i = 0; i < numSlices; i++) {
	    left.insertIntoThis(i * numSlices, 0, slices[i]);
	}

	SimpleMatrix mult = left.mult(right);

	SimpleMatrix result = new SimpleMatrix(numSlices, ncols);
	SimpleMatrix ones = new SimpleMatrix(1, numCols);
	ones.set(1.0);
	for (int i = 0; i < numSlices; i++) {
	    result.insertIntoThis(i, 0, ones.mult(mult.extractMatrix(i*numSlices, i*numSlices + numRows, 0, right.numCols()).elementMult(right)));
	}

	c = 0;
	for (Multiplier request: requests) {
	    request.n.notify(result.extractVector(false, c));
	    c += 1;
	}

	//	System.out.printf("Tensor batch: %d\n", requests.length);
	return requests.length;

    }

    class SCMultiplier {
	public SimpleMatrix right;
	public SimpleMatrix scale;
	public SimpleMatrixNotifier n;
	public int depth;
	public SCMultiplier(SimpleMatrix right, SimpleMatrix scale, SimpleMatrixNotifier n, int depth) {
	    this.right = right;
	    this.scale = scale;
	    this.n = n;
	    this.depth = depth;
	}
	public SCMultiplier(SimpleMatrix right, SimpleMatrix scale, SimpleMatrixNotifier n) {
	    this(right, scale, n, 0);
	}
    }

    transient PriorityQueue<SCMultiplier> scmqueue;

    public void SCMult_async(SimpleMatrix in, SimpleMatrix scale, SimpleMatrixNotifier n) {
	if (scmqueue == null)
	    scmqueue = new PriorityQueue<SCMultiplier>((SCMultiplier left, SCMultiplier right) -> left.depth - right.depth);
	scmqueue.add(new SCMultiplier(in, scale, n));
    }

    transient INDArray leftSCCache = null;
    transient SimpleMatrix simpleLeftSCCache = null;

    void prepGPUSCMult() {
	double[][] leftData = new double[numSlices][];
	for (int i = 0; i < numSlices; i++)
	    leftData[i] = slices[i].getMatrix().data;

	INDArray tmp = Nd4j.create(new int[]{numRows, numCols}, leftData);
	//print_shape(tmp);
	leftSCCache = tmp.addi(tmp.permute(new int[]{0, 2, 1})).reshape(new int[]{numRows, numSlices * numCols}); // TODO : fix the reshape to consider premute()
    }
    
    public int churnSCMultGPU(SCMultiplier[] requests) {
	INDArray left = leftSCCache;

	double[][] rightData = new double[requests.length][];
	double[][] scaleData = new double[requests.length][];
	for (int i = 0; i < requests.length; i++) {
	    rightData[i] = requests[i].right.getMatrix().data;
	    scaleData[i] = requests[i].scale.getMatrix().data;
	}

	INDArray rightT = Nd4j.create(rightData);
	INDArray scaleT = Nd4j.create(scaleData);

	//print_shape(left);
	//print_shape(rightT);
	//print_shape(scaleT);

	INDArray scaledRight = Nd4j.create(numCols * numSlices, requests.length);
	for (int i = 0; i < requests.length; i++) {
	    scaledRight.putColumn(i, rightT.getRow(i).transpose().mmul(scaleT.getRow(i)).linearView()); // TODO: check ordering
	}

	SimpleMatrix result = new SimpleMatrix(numRows, requests.length, true, left.mmul(scaledRight).data().asDouble());

	for (int i = 0; i < requests.length; i++) {
	    requests[i].n.notify(result.extractVector(false, i));
	}

	//	System.out.printf("Tensor batch: %d\n", requests.length);
	return requests.length;
    }

    public int churnSCMult() {
	if (scmqueue == null || scmqueue.size() == 0)
	    return 0;

	SCMultiplier[] requests = scmqueue.toArray(new SCMultiplier[0]);
	scmqueue.clear();

	if (requests.length > 0) {
	    return churnSCMultGPU(requests);
	}
	
	SimpleMatrix left = new SimpleMatrix(numRows, numCols);
	for (int i = 0; i < numSlices; i++) {
	    left = left.plus(getSlice(i).plus(getSlice(i).transpose()));
	}

	SimpleMatrix ones = new SimpleMatrix(requests[0].scale.numRows(), 1);
	ones.set(1.0);

	SimpleMatrix right = new SimpleMatrix(requests[0].right.numRows(), requests.length*requests[0].scale.numRows());

	int c = 0;
	for (SCMultiplier request : requests) {
	    right.insertIntoThis(0, c, request.right.mult(request.scale.transpose()));
	    c += request.scale.numRows();
	}

	SimpleMatrix result = left.mult(right);
	c = 0;
	for (SCMultiplier request: requests) {
	    request.n.notify(result.extractMatrix(0, left.numRows(), c, c + request.scale.numRows()).mult(ones));
	    c += request.scale.numRows();
	}

	//	System.out.printf("Tensor batch: %d\n", requests.length);
	return requests.length;
    }

    public int churn() {
	return churnMult() + (leftCache != null ? churnSCMult() : 0);
    }

    public void prepGPU() {
	prepGPUMult();
	prepGPUSCMult();
    }

  /**
   * Returns true iff every element of the tensor is 0
   */
  public boolean isZero() {
    for (int i = 0; i < numSlices; ++i) {
      if (!NeuralUtils.isZero(slices[i])) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns an iterator over the <code>SimpleMatrix</code> objects contained in the tensor.
   */
  public Iterator<SimpleMatrix> iteratorSimpleMatrix() {
    return Arrays.asList(slices).iterator();
  }

  /**
   * Returns an Iterator which returns the SimpleMatrices represented
   * by an Iterator over tensors.  This is useful for if you want to
   * perform some operation on each of the SimpleMatrix slices, such
   * as turning them into a parameter stack.
   */
  public static Iterator<SimpleMatrix> iteratorSimpleMatrix(Iterator<SimpleTensor> tensors) {
    return new SimpleMatrixIteratorWrapper(tensors);
  }

  private static class SimpleMatrixIteratorWrapper implements Iterator<SimpleMatrix> {
    Iterator<SimpleTensor> tensors;
    Iterator<SimpleMatrix> currentIterator;

    public SimpleMatrixIteratorWrapper(Iterator<SimpleTensor> tensors) {
      this.tensors = tensors;
      advanceIterator();
    }

    public boolean hasNext() {
      if (currentIterator == null) {
        return false;
      }
      if (currentIterator.hasNext()) {
        return true;
      }
      advanceIterator();
      return (currentIterator != null);
    }

    public SimpleMatrix next() {
      if (currentIterator != null && currentIterator.hasNext()) {
        return currentIterator.next();
      }
      advanceIterator();
      if (currentIterator != null) {
        return currentIterator.next();
      }
      throw new NoSuchElementException();
    }

    private void advanceIterator() {
      if (currentIterator != null && currentIterator.hasNext()) {
        return;
      }
      while (tensors.hasNext()) {
        currentIterator = tensors.next().iteratorSimpleMatrix();
        if (currentIterator.hasNext()) {
          return;
        }
      }
      currentIterator = null;
    }

    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    for (int slice = 0; slice < numSlices; ++slice) {
      result.append("Slice " + slice + "\n");
      result.append(slices[slice]);
    }
    return result.toString();
  }

  /**
   * Output the tensor one slice at a time.  Each number is output
   * with the format given, so for example "%f"
   */
  public String toString(String format) {
    StringBuilder result = new StringBuilder();
    for (int slice = 0; slice < numSlices; ++slice) {
      result.append("Slice " + slice + "\n");
      result.append(NeuralUtils.toString(slices[slice], format));
    }
    return result.toString();
  }

  private static final long serialVersionUID = 1;
}
