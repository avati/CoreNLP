package edu.stanford.nlp.sentiment;

import java.util.List;
import java.util.LinkedList;
import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.neural.SimpleTensor;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.ops.factory.DefaultOpFactory;

// TODO: get rid of the word Sentiment everywhere
public class SentimentCostAndGradientGPU extends SentimentCostAndGradient {
  SentimentCostAndGradientGPU(SentimentModel model, List<Tree> trainingBatch) {
    super(model, trainingBatch);
  }

  class BatchedOpInput {
    SimpleMatrix wordVec;
    INDArray nodeVector;
    Tree node;
    Tree root;
    INDArray deltaUp;
    /* constructor for leaf nodes in fwd prop */
    public BatchedOpInput(Tree root, Tree node, SimpleMatrix wordVec) {
      this.root = root;
      this.node = node;
      this.wordVec = wordVec;
    }
    /* constructor for non-leaf nodes in fwd prop */
    public BatchedOpInput(Tree root, Tree node) {
      this.root = root;
      this.node = node;
    }
    /* constructor for classification */
    public BatchedOpInput(Tree node, INDArray nodeVector) {
      this.node = node;
      this.nodeVector = nodeVector;
    }
    /* constructor for nodes in backprop */
    public BatchedOpInput(Tree node, INDArray deltaUp, boolean dummy) {
      this.node = node;
      this.deltaUp = deltaUp;
    }
  };

  List<BatchedOpInput> fwdPropLeaves;
  List<BatchedOpInput> classifyNodes;
  List<BatchedOpInput> fwdPropNonLeaves;
  List<BatchedOpInput> backPropLeaves;
  List<BatchedOpInput> backPropNonLeaves;
  DefaultOpFactory opfactory;

  class SimpleMatrixFake extends SimpleMatrix {
    INDArray matrix;
    public SimpleMatrixFake(INDArray matrix) {
      this.matrix = matrix;
    }
  }

  public void submitFwdPropLeaves(Tree tree, Tree root) {
    if (tree.isLeaf()) {
      throw new AssertionError("We should not have reached leaves in forwardPropagate");
    } else if (tree.children().length > 2) {
      throw new AssertionError("Tree not correctly binarized");
    } else if (tree.isPreTerminal()) {
      String word = tree.children()[0].label().value();
      SimpleMatrix wordVector = model.getWordVector(word);
      fwdPropLeaves.add(new BatchedOpInput(root, tree, wordVector));
    } else if (tree.children().length == 1) {
      throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
    } else {
      submitFwdPropLeaves(tree.children()[0], root);
      submitFwdPropLeaves(tree.children()[1], root);
    }
  }

  void classifyNodes() {
    SimpleMatrix simpleW_s = model.getBinaryClassification("", "");
    INDArray W_sT = Nd4j.create(simpleW_s.getMatrix().data).reshape(simpleW_s.numRows(), simpleW_s.numCols()).transpose();

    BatchedOpInput[] requests = classifyNodes.toArray(new BatchedOpInput[0]);
    classifyNodes.clear();

    INDArray[] perNodeVectors = new INDArray[requests.length];
    for (int i = 0; i < requests.length; i++) {
      perNodeVectors[i] = requests[i].nodeVector;
    }

    INDArray nodeVectors = Nd4j.vstack(perNodeVectors);
    INDArray biases = Nd4j.ones(new int[] {requests.length, 1});

    INDArray softmaxInput = Nd4j.hstack(nodeVectors, biases).mmul(W_sT);
    INDArray predictions = Nd4j.create(softmaxInput.shape());
    Nd4j.getExecutioner().exec(opfactory.createTransform("softmax", softmaxInput, predictions));

    double[] indices = Nd4j.argMax(predictions, 1).data().asDouble();

    for (int i = 0; i < requests.length; i++) {
      CoreLabel label = (CoreLabel) requests[i].node.label();
      label.set(RNNCoreAnnotations.Predictions.class, new SimpleMatrixFake(predictions.getRow(i)));
      label.set(RNNCoreAnnotations.PredictedClass.class, (int)indices[i]);
    }
  }

  INDArray oneMinusXSquared(INDArray x) {
    return x.muli(x, Nd4j.create(x.shape())).subi(1).muli(-1);
  }

  void processFwdPropCommon(INDArray tanhInput, BatchedOpInput[] requests) {
    INDArray nodeVectors = Nd4j.create(tanhInput.shape());
    Nd4j.getExecutioner().exec(opfactory.createTransform("tanh", tanhInput, nodeVectors));
    //    INDArray nodeVectors = Transforms.tanh(tanhInput);
    INDArray nodeVectorDerivatives = oneMinusXSquared(nodeVectors);

    for (int i = 0; i < requests.length; i++) {
      CoreLabel label = (CoreLabel) requests[i].node.label();
      Tree root = requests[i].root;
      Tree node = requests[i].node;
      Tree parent = node.parent(root);
      INDArray nodeVector = nodeVectors.getRow(i);
      INDArray nodeVectorDerivative = nodeVectorDerivatives.getRow(i);
      label.set(RNNCoreAnnotations.NodeVector.class, new SimpleMatrixFake(nodeVector));
      label.set(RNNCoreAnnotations.NodeVectorDerivative.class, new SimpleMatrixFake(nodeVectorDerivative));

      classifyNodes.add(new BatchedOpInput(node, nodeVector));

      if (parent == null ||
	  RNNCoreAnnotations.getNodeVector(parent.children()[0]) == null ||
	  RNNCoreAnnotations.getNodeVector(parent.children()[1]) == null)
	continue;
      fwdPropNonLeaves.add(new BatchedOpInput(root, parent));
    }
  }

  void processFwdPropLeaves() {
    BatchedOpInput[] requests = fwdPropLeaves.toArray(new BatchedOpInput[0]);
    fwdPropLeaves.clear();

    double[][] wordVecData = new double[requests.length][];
    for (int i = 0; i < requests.length; i++)
      wordVecData[i] = requests[i].wordVec.getMatrix().data;

    processFwdPropCommon(Nd4j.create(wordVecData), requests);
  }

  void processFwdPropNonLeaves() {
    //SimpleMatrix simpleW_s = model.getUnaryClassification(tree.label().value());
    SimpleMatrix simpleW = model.getBinaryTransform("", "");
    INDArray WT = Nd4j.create(simpleW.getMatrix().data).reshape(simpleW.numRows(), simpleW.numCols()).transpose();

    while (fwdPropNonLeaves.size() > 0) {
      BatchedOpInput[] requests = fwdPropNonLeaves.toArray(new BatchedOpInput[0]);
      fwdPropNonLeaves.clear();

      INDArray[] leftNodeVectors = new INDArray[requests.length];
      INDArray[] rightNodeVectors = new INDArray[requests.length];
      for (int i = 0; i < requests.length; i++) {
	leftNodeVectors[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getNodeVector(requests[i].node.children()[0])).matrix;
	rightNodeVectors[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getNodeVector(requests[i].node.children()[1])).matrix;
      }
      INDArray biases = Nd4j.ones(new int[] {requests.length, 1});
      INDArray inputVectors = Nd4j.hstack(Nd4j.vstack(leftNodeVectors), Nd4j.vstack(rightNodeVectors));
      INDArray childrenVectors = Nd4j.hstack(inputVectors, biases);
      
      INDArray tanhInput = childrenVectors.mmul(WT);

      if (model.op.useTensors) {
	INDArray blp = bilinearProductsGPU(inputVectors);
	tanhInput = tanhInput.addi(blp);
      }

      for (int i = 0; i < requests.length; i++) {
	CoreLabel label = (CoreLabel) requests[i].node.label();
	label.set(RNNCoreAnnotations.InputVector.class, new SimpleMatrixFake(inputVectors.getRow(i)));
      }

      processFwdPropCommon(tanhInput, requests);
    }
  }

  double normF(INDArray arr) {
    return arr.distance2(Nd4j.zeros(arr.shape()));
  }

  INDArray rowTensorT = null;
  INDArray flatSymmetrizedTensor = null;

  INDArray khatriRaoProductRowWise(INDArray left, INDArray right) {
    int leftRows = left.size(0);
    int rightRows = right.size(0);
    int leftCols = left.size(1);
    int rightCols = right.size(1);

    return Nd4j.tile(left.transpose(), 1, rightCols).reshape(rightCols*leftCols, leftRows).transpose().muli(Nd4j.tile(right, 1, leftCols));
  }


  INDArray khatriRaoProduct(INDArray left, INDArray right) {
    int leftRows = left.size(0);
    int rightRows = right.size(0);
    int leftCols = left.size(1);
    int rightCols = right.size(1);

    return Nd4j.tile(left, 1, rightRows).reshape(leftRows*rightRows, leftCols).muli(Nd4j.tile(right, leftRows, 1));
  }

  INDArray bilinearProductsGPU(INDArray tensorIn) {
    return khatriRaoProductRowWise(tensorIn, tensorIn).mmul(rowTensorT);
  }

  INDArray[] goldLabels;
  INDArray zeroLabel;
  
  void processBackPropCommon(BatchedOpInput[] requests, ModelDerivatives derivatives, INDArray W, INDArray WsT, boolean isPreTerminal) {
    INDArray[] predictionsArray = new INDArray[requests.length];
    INDArray[] nodeVectorsArray = new INDArray[requests.length];
    INDArray[] nodeVectorDerivativesArray = new INDArray[requests.length];
    INDArray[] goldLabelsArray = new INDArray[requests.length];
    INDArray[] deltaUpsArray = new INDArray[requests.length];
    double[] nodeWeightsArray = new double[requests.length];
    INDArray biases = Nd4j.ones(new int[]{requests.length, 1});

    for (int i = 0; i < requests.length; i++) {
      nodeVectorsArray[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getNodeVector(requests[i].node)).matrix;
      nodeVectorDerivativesArray[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getNodeVectorDerivative(requests[i].node)).matrix;
      predictionsArray[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getPredictions(requests[i].node)).matrix;
      deltaUpsArray[i] = requests[i].deltaUp;
      int goldClass = RNNCoreAnnotations.getGoldClass(requests[i].node);
      if (goldClass >= 0) {
	goldLabelsArray[i] = goldLabels[goldClass];
	nodeWeightsArray[i] = model.op.trainOptions.getClassWeight(goldClass);
      } else {
	goldLabelsArray[i] = zeroLabel;
	nodeWeightsArray[i] = 0;
      }
    }

    INDArray predictions = Nd4j.vstack(predictionsArray);
    INDArray golds = Nd4j.vstack(goldLabelsArray);
    INDArray nodeWeights = Nd4j.create(nodeWeightsArray).transpose();
    INDArray nodeVectors = Nd4j.vstack(nodeVectorsArray);
    INDArray nodeVectorDerivatives = Nd4j.vstack(nodeVectorDerivativesArray);
    INDArray deltaUps = Nd4j.vstack(deltaUpsArray);

    INDArray deltaClass = predictions.sub(golds).muli(Nd4j.tile(nodeWeights, 1, model.numClasses)).transpose();
    INDArray localCD = deltaClass.mmul(Nd4j.hstack(Nd4j.vstack(nodeVectorsArray), biases));

    SimpleMatrix simpleLocalCD = new SimpleMatrix(localCD.shape()[0], localCD.shape()[1], true, localCD.data().asDouble()); // TODO verify ordering
    derivatives.unaryCD.put("", derivatives.unaryCD.get("").plus(simpleLocalCD));

    double[] errors = Transforms.log(predictions).muli(golds).sum(1).muli(nodeWeights).muli(-1).data().asDouble();

    for (int i = 0; i < requests.length; i++) {
      RNNCoreAnnotations.setPredictionError(requests[i].node, errors[i]);
    }

    int[] rowNums = new int[model.op.numHid];
    for (int i = 0; i < model.op.numHid; i++)
      rowNums[i] = i;
    INDArray deltaFulls = WsT.mmul(deltaClass).getRows(rowNums).transpose().muli(nodeVectorDerivatives).addi(deltaUps);

    if (isPreTerminal) {
      SimpleMatrix simpleDeltaFulls = new SimpleMatrix(model.op.numHid, requests.length, false, deltaFulls.data().asDouble()); //TODO verify ordering
      for (int i = 0; i < requests.length; i++) {
	String word = model.getVocabWord(requests[i].node.children()[0].label().value());
	SimpleMatrix oldWordVectorD = derivatives.wordVectorD.get(word);
	SimpleMatrix deltaFull = simpleDeltaFulls.extractVector(false, i);
	if (oldWordVectorD == null) {
	  derivatives.wordVectorD.put(word, deltaFull);
	} else {
	  derivatives.wordVectorD.put(word, deltaFull);
	}
      }
      return;
    }


    INDArray[] inputVectorsArray = new INDArray[requests.length];
    INDArray[] leftDerivativesArray = new INDArray[requests.length];
    INDArray[] rightDerivativesArray = new INDArray[requests.length];
    for (int i = 0; i < requests.length; i++) {
      inputVectorsArray[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getInputVector(requests[i].node)).matrix;
      leftDerivativesArray[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getNodeVectorDerivative(requests[i].node.children()[0])).matrix;
      rightDerivativesArray[i] = ((SimpleMatrixFake)RNNCoreAnnotations.getNodeVectorDerivative(requests[i].node.children()[1])).matrix;
    }
    INDArray inputVectors = Nd4j.vstack(inputVectorsArray);
    INDArray childrenVectors = Nd4j.hstack(inputVectors, biases);
    INDArray leftDerivatives = Nd4j.vstack(leftDerivativesArray);
    INDArray rightDerivatives = Nd4j.vstack(rightDerivativesArray);

    INDArray W_df = deltaFulls.transpose().mmul(childrenVectors);
    SimpleMatrix simpleW_df = new SimpleMatrix(W_df.shape()[0], W_df.shape()[1], true, W_df.data().asDouble()); // TODO verify ordering
    derivatives.binaryTD.put("", "", derivatives.binaryTD.get("", "").plus(simpleW_df));


    INDArray deltaDowns;
    if (model.op.useTensors) {
      SimpleTensor Wt_df = getTensorGradientGPU(deltaFulls, inputVectors);
      derivatives.binaryTensorTD.put("", "", derivatives.binaryTensorTD.get("", "").plus(Wt_df));
      deltaDowns = computeTensorDeltaDownsGPU(deltaFulls, inputVectors, W);
    } else {
      deltaDowns = deltaFulls.mmul(W);
    }

    int[] leftCols = new int[deltaFulls.shape()[1]];
    for (int i = 0; i < leftCols.length; i++)
      leftCols[i] = i;
    int[] rightCols = new int[deltaFulls.shape()[1]];
    for (int i = 0; i < rightCols.length; i++)
      rightCols[i] = i + leftCols.length;
    INDArray leftDeltaDowns = deltaDowns.getColumns(leftCols);
    INDArray rightDeltaDowns = deltaDowns.getColumns(rightCols);
    INDArray leftDeltaUps = leftDeltaDowns.muli(leftDerivatives);
    INDArray rightDeltaUps = rightDeltaDowns.muli(rightDerivatives);

    for (int i = 0; i < requests.length; i++) {
      Tree node = requests[i].node;
      Tree left = node.children()[0];
      Tree right = node.children()[1];

      if (left.isPreTerminal())
	backPropLeaves.add(new BatchedOpInput(left, leftDeltaUps.getRow(i), true));
      else
	backPropNonLeaves.add(new BatchedOpInput(left, leftDeltaUps.getRow(i), true));

      if (right.isPreTerminal())
	backPropLeaves.add(new BatchedOpInput(right, rightDeltaUps.getRow(i), true));
      else
	backPropNonLeaves.add(new BatchedOpInput(right, rightDeltaUps.getRow(i), true));
    }
  }

  INDArray computeTensorDeltaDownsGPU(INDArray deltaFulls, INDArray inputVectors, INDArray W) {
    int numRows = inputVectors.size(1);
    int numSlices = deltaFulls.size(1);
    int batchSize = deltaFulls.size(0);
    
    int[] rowNums = new int[numSlices*2];
    for (int i = 0; i < rowNums.length; i++)
      rowNums[i] = i;
    INDArray WTDeltaNoBias = deltaFulls.mmul(W).getColumns(rowNums);

    INDArray deltaDowns = khatriRaoProductRowWise(deltaFulls, inputVectors).mmul(flatSymmetrizedTensor);
    return WTDeltaNoBias.addi(deltaDowns);
  }

  SimpleTensor getTensorGradientGPU(INDArray deltaFulls, INDArray inputVectors) {
    int numRows = inputVectors.size(1);
    int numSlices = deltaFulls.size(1);
    int batchSize = deltaFulls.size(0);
    INDArray WtFlat_df = inputVectors.transpose().mmul(khatriRaoProductRowWise(deltaFulls, inputVectors));
    double[] flatWt_df = WtFlat_df.data().asDouble();
    SimpleMatrix[] slices = new SimpleMatrix[numSlices];
    for (int i = 0; i < numSlices; i++) {
      slices[i] = new SimpleMatrix(numRows, numRows, true, Arrays.copyOfRange(flatWt_df, numRows*numRows*i, numRows*numRows*(i+1)));
    }
    return new SimpleTensor(slices);
  }

  void processBackPropNonLeaves(ModelDerivatives derivatives) {
    SimpleMatrix simpleW = model.getBinaryTransform("", "");
    INDArray W = Nd4j.create(simpleW.getMatrix().data).reshape(simpleW.numRows(), simpleW.numCols());
    SimpleMatrix simpleWs = model.getBinaryClassification("", "");
    INDArray WsT = Nd4j.create(simpleWs.getMatrix().data).reshape(simpleWs.numRows(), simpleWs.numCols()).transpose();
	
    while (backPropNonLeaves.size() > 0) {
      BatchedOpInput[] requests = backPropNonLeaves.toArray(new BatchedOpInput[0]);
      backPropNonLeaves.clear();

      processBackPropCommon(requests, derivatives, W, WsT, false);
    }
  }

  void processBackPropLeaves(ModelDerivatives derivatives) {
    SimpleMatrix simpleWs = model.getUnaryClassification("");
    INDArray WsT = Nd4j.create(simpleWs.getMatrix().data).reshape(simpleWs.numRows(), simpleWs.numCols()).transpose();
	
    BatchedOpInput[] requests = backPropLeaves.toArray(new BatchedOpInput[0]);
    backPropLeaves.clear();

    processBackPropCommon(requests, derivatives, null, WsT, true);
  }

  void flattenTensors() {
    SimpleTensor t = model.getBinaryTensor("", "");
    double[][] tensorData = new double[t.numSlices()][];
    double[][] tensorTData = new double[t.numSlices()][];
    for (int i = 0; i < t.numSlices(); i++) {
      tensorData[i] = t.getSlice(i).getMatrix().data;
      tensorTData[i] = t.getSlice(i).transpose().getMatrix().data;
    }

    rowTensorT = Nd4j.create(tensorData).transpose();
    
    flatSymmetrizedTensor = Nd4j.create(tensorData).addi(Nd4j.create(tensorTData)).transpose().reshape(t.numRows(), t.numSlices()*t.numCols()).transpose();
  }

  void prepGPU() {
    if (fwdPropLeaves == null) fwdPropLeaves = new LinkedList<BatchedOpInput>();
    if (classifyNodes == null) classifyNodes = new LinkedList<BatchedOpInput>();
    if (fwdPropNonLeaves == null) fwdPropNonLeaves = new LinkedList<BatchedOpInput>();
    if (backPropLeaves == null) backPropLeaves = new LinkedList<BatchedOpInput>();
    if (backPropNonLeaves == null) backPropNonLeaves = new LinkedList<BatchedOpInput>();
    
    if (opfactory == null) opfactory = new DefaultOpFactory();

    if (model.op.useTensors)
      flattenTensors();

    if (goldLabels == null) {
      goldLabels = new INDArray[model.numClasses];
      for (int i = 0; i < model.numClasses; i++) {
	goldLabels[i] = Nd4j.create(model.numClasses);
	goldLabels[i].put(0, i, 1.0);
      }
      zeroLabel = Nd4j.create(model.numClasses);
    }
  }

  protected ModelDerivatives scoreDerivatives(List<Tree> trainingBatch) {
    // "final" makes this as fast as having separate maps declared in this function
    final ModelDerivatives derivatives = new ModelDerivatives(model);

    prepGPU();

    List<Tree> forwardPropTrees = Generics.newArrayList();
    for (Tree tree : trainingBatch) {
      Tree trainingTree = tree.deepCopy();
      // this will attach the error vectors and the node vectors
      // to each node in the tree
      submitFwdPropLeaves(trainingTree, trainingTree);
      forwardPropTrees.add(trainingTree);
    }

    processFwdPropLeaves();
    processFwdPropNonLeaves();
    classifyNodes();
    
    for (Tree tree : forwardPropTrees) {
      INDArray deltaUp = Nd4j.create(model.op.numHid);
      backPropNonLeaves.add(new BatchedOpInput(tree, deltaUp, true));
    }

    processBackPropNonLeaves(derivatives);
    processBackPropLeaves(derivatives);

    
    for (Tree tree : forwardPropTrees) {
      derivatives.error += sumError(tree);
    }

    return derivatives;
  }
}
