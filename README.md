package beast.evolution.likelihood;
import jdk.incubator.vector.*;

/**
 * standard likelihood core, uses no caching *
 */
public class BeerLikelihoodCore extends LikelihoodCore {
    protected int nrOfStates;
    protected int nrOfNodes;
    protected int nrOfPatterns;
    protected int partialsSize;
    protected int matrixSize;
    protected int nrOfMatrices;

    protected boolean integrateCategories;

    protected double[][][] partials;

    protected int[][] states;

    protected double[][][] matrices;

    protected int[] currentMatrixIndex;
    protected int[] storedMatrixIndex;
    protected int[] currentPartialsIndex;
    protected int[] storedPartialsIndex;

    protected boolean useScaling = false;

    protected double[][][] scalingFactors;

    private double scalingThreshold = 1.0E-100;
    double SCALE = 2;
    protected VectorSpecies<Double> SPECIES;
    protected int upperBound;
    public BeerLikelihoodCore(int nrOfStates) {

        this.nrOfStates = nrOfStates;
        this.SPECIES = DoubleVector.SPECIES_256;
        this.upperBound = SPECIES.loopBound(this.nrOfStates);
    } // c'tor


    /**
     * Calculates partial likelihoods at a node when both children have states.
     */
    protected void calculateStatesStatesPruning(int[] stateIndex1, double[] matrices1,
                                                int[] stateIndex2, double[] matrices2,
                                                double[] partials3) {
        int v = 0;

        for (int l = 0; l < nrOfMatrices; l++) {

            for (int k = 0; k < nrOfPatterns; k++) {

                int state1 = stateIndex1[k];
                int state2 = stateIndex2[k];

                int w = l * matrixSize;

                if (state1 < nrOfStates && state2 < nrOfStates) {
                    int i = 0;
                    for(;i<upperBound;i+=SPECIES.length(),w+=nrOfStates,v++){
                        DoubleVector vp = DoubleVector.fromArray(SPECIES, partials3, v);
                        DoubleVector vm1 = DoubleVector.fromArray(SPECIES, matrices1, w + state1);
                        DoubleVector vm2 = DoubleVector.fromArray(SPECIES, matrices2, w + state2);
                        vp=vp.sub(vp).add(vm1.mul(vm2));
                        vp.intoArray(partials3, v);
                    }
                    for(;i<nrOfStates;i++,w+=nrOfStates,v++){
                        partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                    }

                } else if (state1 < nrOfStates) {
                    // child 2 has a gap or unknown state so treat it as unknown

                    int i = 0;
                    for(;i<upperBound;i+=SPECIES.length(),w+=nrOfStates,v++){
                        DoubleVector vp = DoubleVector.fromArray(SPECIES, partials3, v);
                        DoubleVector vm1 = DoubleVector.fromArray(SPECIES, matrices1, w + state1);
                        vp=vp.sub(vp).add(vm1);
                        vp.intoArray(partials3, v);
                    }
                    for(;i<nrOfStates;i++,w+=nrOfStates,v++){
                        partials3[v] = matrices1[w + state1];
                    }

                } else if (state2 < nrOfStates) {
                    // child 2 has a gap or unknown state so treat it as unknown

                    int i = 0;
                    for(;i<upperBound;i+=SPECIES.length(),w+=nrOfStates,v++){
                        DoubleVector vp = DoubleVector.fromArray(SPECIES, partials3, v);
                        DoubleVector vm2 = DoubleVector.fromArray(SPECIES, matrices2, w + state2);
                        vp=vp.sub(vp).add(vm2);
                        vp.intoArray(partials3, v);
                    }
                    for(;i<nrOfStates;i++,w+=nrOfStates,v++){
                        partials3[v] = matrices2[w + state2];
                    }

                } else {
                    // both children have a gap or unknown state so set partials to 1
                    int j = 0;
                    for(;j<upperBound;j+=SPECIES.length(),w+=nrOfStates,v++){
                        DoubleVector vm = DoubleVector.fromArray(SPECIES, partials3, v);
                        vm=vm.sub(vm).add(1);
                        vm.intoArray(partials3, v);
                    }
                    for(;j<nrOfStates;j++,w+=nrOfStates,v++) {
                        partials3[v] = 1.0;
                    }
                }
            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when one child has states and one has partials.
     */
    protected void calculateStatesPartialsPruning(int[] stateIndex1, double[] matrices1,
                                                  double[] partials2, double[] matrices2,
                                                  double[] partials3) {

        double tmp;
        double final_sum;
        DoubleVector sum;

        int u = 0;
        int v = 0;

        for (int l = 0; l < nrOfMatrices; l++) {
            for (int k = 0; k < nrOfPatterns; k++) {

                int state1 = stateIndex1[k];

                int w = l * matrixSize;

                if (state1 < nrOfStates) {


                    for (int i = 0; i < nrOfStates; i++) {

                        tmp = matrices1[w + state1];

                        sum = DoubleVector.zero(SPECIES);
                        int j = 0;
                        for (; j < upperBound;j+=SPECIES.length(),w++) {

                            DoubleVector vp = DoubleVector.fromArray(SPECIES, partials2, v + j);
                            DoubleVector vm = DoubleVector.fromArray(SPECIES, matrices2, w);
                            sum = vp.fma(vm, sum);
                        }
                        final_sum = sum.reduceLanes(VectorOperators.ADD);
                        for (; j < nrOfStates; j++,w++) {
                            final_sum += partials2[v + j] * matrices2[w];
                        }
                        partials3[u] = tmp * final_sum;
                        u++;
                    }

                    v += nrOfStates;
                } else {
                    // Child 1 has a gap or unknown state so don't use it

                    for (int i = 0; i < nrOfStates; i++) {

                        sum = DoubleVector.zero(SPECIES);
                        int j = 0;
                        for (; j < upperBound;j+=SPECIES.length(),w++) {

                            DoubleVector vp = DoubleVector.fromArray(SPECIES, partials2, v + j);
                            DoubleVector vm = DoubleVector.fromArray(SPECIES, matrices2, w);
                            sum = vp.fma(vm, sum);
                        }
                        final_sum = sum.reduceLanes(VectorOperators.ADD);
                        for (; j < nrOfStates; j++,w++) {
                            final_sum += partials2[v + j] * matrices2[w];
                        }
                        partials3[u] = final_sum;
                        u++;
                    }

                    v += nrOfStates;
                }
            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when both children have partials.
     */
    protected void calculatePartialsPartialsPruning(double[] partials1, double[] matrices1,
                                                    double[] partials2, double[] matrices2,
                                                    double[] partials3) {

        double final_sum1,final_sum2;
        DoubleVector sum1,sum2;

        int u = 0;
        int v = 0;

        for (int l = 0; l < nrOfMatrices; l++) {

            for (int k = 0; k < nrOfPatterns; k++) {

                int w = l * matrixSize;

                for (int i = 0; i < nrOfStates; i++) {

                    sum1 = sum2 = DoubleVector.zero(SPECIES);
                    int j = 0;
                    for (; j < upperBound;j+=SPECIES.length(),w++) {

                        DoubleVector vp1 = DoubleVector.fromArray(SPECIES, partials1, v + j);
                        DoubleVector vm1 = DoubleVector.fromArray(SPECIES, matrices1, w);
                        DoubleVector vp2 = DoubleVector.fromArray(SPECIES, partials2, v + j);
                        DoubleVector vm2 = DoubleVector.fromArray(SPECIES, matrices2, w);
                        sum1 = vp1.fma(vm1, sum1);
                        sum2 = vp2.fma(vm2, sum2);
                    }
                    final_sum1= sum1.reduceLanes(VectorOperators.ADD);
                    final_sum2 = sum2.reduceLanes(VectorOperators.ADD);
                    for (; j < nrOfStates; j++,w++) {
                        final_sum1 += matrices1[w] * partials1[v + j];
                        final_sum2 += matrices2[w] * partials2[v + j];
                    }
                    partials3[u] = final_sum1 * final_sum2;
                    u++;
                }
                v += nrOfStates;
            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when both children have states.
     */
    protected void calculateStatesStatesPruning(int[] stateIndex1, double[] matrices1,
                                                int[] stateIndex2, double[] matrices2,
                                                double[] partials3, int[] matrixMap) {
        int v = 0;

        for (int k = 0; k < nrOfPatterns; k++) {

            int state1 = stateIndex1[k];
            int state2 = stateIndex2[k];

            int w = matrixMap[k] * matrixSize;

            if (state1 < nrOfStates && state2 < nrOfStates) {

                int i = 0;
                for(;i<upperBound;i+=SPECIES.length(),w+=nrOfStates,v++){
                    DoubleVector vp = DoubleVector.fromArray(SPECIES, partials3, v);
                    DoubleVector vm1 = DoubleVector.fromArray(SPECIES, matrices1, w + state1);
                    DoubleVector vm2 = DoubleVector.fromArray(SPECIES, matrices2, w + state2);
                    vp=vp.sub(vp).add(vm1.mul(vm2));
                    vp.intoArray(partials3, v);
                }
                for(;i<nrOfStates;i++,w+=nrOfStates,v++){
                    partials3[v] = matrices1[w + state1] * matrices2[w + state2];
                }



            } else if (state1 < nrOfStates) {
                // child 2 has a gap or unknown state so treat it as unknown

                int i = 0;
                for(;i<upperBound;i+=SPECIES.length(),w+=nrOfStates,v++){
                    DoubleVector vp = DoubleVector.fromArray(SPECIES, partials3, v);
                    DoubleVector vm1 = DoubleVector.fromArray(SPECIES, matrices1, w + state1);
                    vp=vp.sub(vp).add(vm1);
                    vp.intoArray(partials3, v);
                }
                for(;i<nrOfStates;i++,w+=nrOfStates,v++){
                    partials3[v] = matrices1[w + state1];
                }

            } else if (state2 < nrOfStates) {
                // child 2 has a gap or unknown state so treat it as unknown

                int i = 0;
                for(;i<upperBound;i+=SPECIES.length(),w+=nrOfStates,v++){
                    DoubleVector vp = DoubleVector.fromArray(SPECIES, partials3, v);
                    DoubleVector vm2 = DoubleVector.fromArray(SPECIES, matrices2, w + state2);
                    vp=vp.add(vm2);
                    vp.sub(vp).intoArray(partials3, v);
                }
                for(;i<nrOfStates;i++,w+=nrOfStates,v++){
                    partials3[v] = matrices2[w + state2];
                }
            } else {
                // both children have a gap or unknown state so set partials to 1
                int j = 0;
                for(;j<upperBound;j+=SPECIES.length(),w+=nrOfStates,v++){
                    DoubleVector vm = DoubleVector.fromArray(SPECIES, partials3, v);
                    vm=vm.sub(vm).add(1);
                    vm.intoArray(partials3, v);
                }
                for(;j<nrOfStates;j++,w+=nrOfStates,v++) {
                    partials3[v] = 1.0;
                }
            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when one child has states and one has partials.
     */
    protected void calculateStatesPartialsPruning(int[] stateIndex1, double[] matrices1,
                                                  double[] partials2, double[] matrices2,
                                                  double[] partials3, int[] matrixMap) {

        double tmp;
        double final_sum;
        DoubleVector sum;

        int u = 0;
        int v = 0;

        for (int k = 0; k < nrOfPatterns; k++) {

            int state1 = stateIndex1[k];

            int w = matrixMap[k] * matrixSize;

            if (state1 < nrOfStates) {

                for (int i = 0; i < nrOfStates; i++) {


                    tmp = matrices1[w + state1];
                    sum = DoubleVector.zero(SPECIES);
                    int j = 0;
                    for (; j < upperBound;j+=SPECIES.length(),w++) {
                        DoubleVector vp = DoubleVector.fromArray(SPECIES, partials2, v + j);
                        DoubleVector vm = DoubleVector.fromArray(SPECIES, matrices2, w);
                        sum = vp.fma(vm, sum);
                    }
                    final_sum = sum.reduceLanes(VectorOperators.ADD);
                    for (; j < nrOfStates; j++,w++) {
                        final_sum += partials2[v + j] * matrices2[w];
                    }
                    partials3[u] = tmp * final_sum;
                    u++;
                }

                v += nrOfStates;
            } else {
                // Child 1 has a gap or unknown state so don't use it

                for (int i = 0; i < nrOfStates; i++) {

                    sum = DoubleVector.zero(SPECIES);
                    int j = 0;
                    for (; j < upperBound;j+=SPECIES.length(),w++) {

                        DoubleVector vp = DoubleVector.fromArray(SPECIES, partials2, v + j);
                        DoubleVector vm = DoubleVector.fromArray(SPECIES, matrices2, w);
                        sum = vp.fma(vm, sum);
                    }
                    final_sum = sum.reduceLanes(VectorOperators.ADD);
                    for (; j < nrOfStates; j++,w++) {
                        final_sum += partials2[v + j] * matrices2[w];
                    }
                    partials3[u] = final_sum;
                    u++;
                }

                v += nrOfStates;
            }
        }
    }

    /**
     * Calculates partial likelihoods at a node when both children have partials.
     */
    protected void calculatePartialsPartialsPruning(double[] partials1, double[] matrices1,
                                                    double[] partials2, double[] matrices2,
                                                    double[] partials3, int[] matrixMap) {
        double final_sum1,final_sum2;
        DoubleVector sum1,sum2;

        int u = 0;
        int v = 0;

        for (int k = 0; k < nrOfPatterns; k++) {

            int w = matrixMap[k] * matrixSize;

            for (int i = 0; i < nrOfStates; i++) {

                sum1 = sum2 = DoubleVector.zero(SPECIES);
                int j = 0;
                for (; j < upperBound;j+=SPECIES.length(),w++) {

                    DoubleVector vp1 = DoubleVector.fromArray(SPECIES, partials1, v + j);
                    DoubleVector vm1 = DoubleVector.fromArray(SPECIES, matrices1, w);
                    DoubleVector vp2 = DoubleVector.fromArray(SPECIES, partials2, v + j);
                    DoubleVector vm2 = DoubleVector.fromArray(SPECIES, matrices2, w);
                    sum1 = vp1.fma(vm1, sum1);
                    sum2 = vp2.fma(vm2, sum2);
                }
                final_sum1= sum1.reduceLanes(VectorOperators.ADD);
                final_sum2 = sum2.reduceLanes(VectorOperators.ADD);
                for (; j < nrOfStates; j++,w++) {
                    final_sum1 += matrices1[w] * partials1[v + j];
                    final_sum2 += matrices2[w] * partials2[v + j];
                }
                partials3[u] = final_sum1 * final_sum2;
                u++;
            }
            v += nrOfStates;
        }
    }

    /**
     * Integrates partials across categories.
     *
     * @param inPartials  the array of partials to be integrated
     * @param proportions the proportions of sites in each category
     * @param outPartials an array into which the partials will go
     */
    @Override
    protected void calculateIntegratePartials(double[] inPartials, double[] proportions, double[] outPartials) {

        int u = 0;
        int v = 0;
        for (int k = 0; k < nrOfPatterns; k++) {
            int i = 0;
            for(;i<upperBound;i+=SPECIES.length(),u++,v++){
                DoubleVector vop = DoubleVector.fromArray(SPECIES, outPartials, u);
                DoubleVector vinp = DoubleVector.fromArray(SPECIES, inPartials, v);
                DoubleVector vp = DoubleVector.fromArray(SPECIES, proportions, 0);
                vop=vop.sub(vop).add(vinp.mul(vp));
                vop.intoArray(outPartials, u);
            }
            for (; i < nrOfStates; i++,u++,v++) {
                outPartials[u] = inPartials[v] * proportions[0];
            }
        }


        for (int l = 1; l < nrOfMatrices; l++) {
            u = 0;

            for (int k = 0; k < nrOfPatterns; k++) {

                int i = 0;
                for(;i<upperBound;i+=SPECIES.length(),u++,v++){
                    DoubleVector vop = DoubleVector.fromArray(SPECIES, outPartials, u);
                    DoubleVector vinp = DoubleVector.fromArray(SPECIES, inPartials, v);
                    DoubleVector vp = DoubleVector.fromArray(SPECIES, proportions, l);
                    vop.add(vinp.mul(vp));
                    vop.intoArray(outPartials, u);
                }
                for (; i < nrOfStates; i++,u++,v++) {
                    outPartials[u] += inPartials[v] * proportions[l];
                }
            }
        }
    }

    /**
     * Calculates pattern log likelihoods at a node.
     *
     * @param partials          the partials used to calculate the likelihoods
     * @param frequencies       an array of state frequencies
     * @param outLogLikelihoods an array into which the likelihoods will go
     */
    @Override
    public void calculateLogLikelihoods(double[] partials, double[] frequencies, double[] outLogLikelihoods) {
        int v = 0;
        double final_sum1;
        DoubleVector sum1;
        for (int k = 0; k < nrOfPatterns; k++) {
            sum1 = DoubleVector.zero(SPECIES);
            int i = 0;
            for (; i < upperBound;i+=SPECIES.length(),v++) {

                DoubleVector vf = DoubleVector.fromArray(SPECIES, frequencies, i);
                DoubleVector vp = DoubleVector.fromArray(SPECIES, partials, v);
                sum1 = vf.fma(vp, sum1);
            }
            final_sum1= sum1.reduceLanes(VectorOperators.ADD);
            for (; i < nrOfStates; i++,v++) {
                final_sum1 += frequencies[i] * partials[v];
            }
            outLogLikelihoods[k] = Math.log(final_sum1) + getLogScalingFactor(k);
        }
    }


    /**
     * initializes partial likelihood arrays.
     *
     * @param nodeCount           the number of nodes in the tree
     * @param patternCount        the number of patterns
     * @param matrixCount         the number of matrices (i.e., number of categories)
     * @param integrateCategories whether sites are being integrated over all matrices
     */
    @Override
    public void initialize(int nodeCount, int patternCount, int matrixCount, boolean integrateCategories, boolean useAmbiguities) {

        this.nrOfNodes = nodeCount;
        this.nrOfPatterns = patternCount;
        this.nrOfMatrices = matrixCount;

        this.integrateCategories = integrateCategories;

        if (integrateCategories) {
            partialsSize = patternCount * nrOfStates * matrixCount;
        } else {
            partialsSize = patternCount * nrOfStates;
        }

        partials = new double[2][nodeCount][];

        currentMatrixIndex = new int[nodeCount];
        storedMatrixIndex = new int[nodeCount];

        currentPartialsIndex = new int[nodeCount];
        storedPartialsIndex = new int[nodeCount];

        states = new int[nodeCount][];

        for (int i = 0; i < nodeCount; i++) {
            partials[0][i] = null;
            partials[1][i] = null;

            states[i] = null;
        }

        matrixSize = nrOfStates * nrOfStates;

        matrices = new double[2][nodeCount][matrixCount * matrixSize];
    }

    /**
     * cleans up and deallocates arrays.
     */
    @Override
    public void finalize() throws java.lang.Throwable {
        nrOfNodes = 0;
        nrOfPatterns = 0;
        nrOfMatrices = 0;

        partials = null;
        currentPartialsIndex = null;
        storedPartialsIndex = null;
        states = null;
        matrices = null;
        currentMatrixIndex = null;
        storedMatrixIndex = null;

        scalingFactors = null;
    }

    @Override
    public void setUseScaling(double scale) {
        useScaling = (scale != 1.0);

        if (useScaling) {
            scalingFactors = new double[2][nrOfNodes][nrOfPatterns];
        }
    }

    /**
     * Allocates partials for a node
     */
    @Override
    public void createNodePartials(int nodeIndex) {

        this.partials[0][nodeIndex] = new double[partialsSize];
        this.partials[1][nodeIndex] = new double[partialsSize];
    }

    /**
     * Sets partials for a node
     */
    @Override
    public void setNodePartials(int nodeIndex, double[] partials) {

        if (this.partials[0][nodeIndex] == null) {
            createNodePartials(nodeIndex);
        }
        if (partials.length < partialsSize) {
            int k = 0;
            for (int i = 0; i < nrOfMatrices; i++) {
                System.arraycopy(partials, 0, this.partials[0][nodeIndex], k, partials.length);
                k += partials.length;
            }
        } else {
            System.arraycopy(partials, 0, this.partials[0][nodeIndex], 0, partials.length);
        }
    }

    @Override
    public void getNodePartials(int nodeIndex, double[] partialsOut) {
        System.arraycopy(partials[currentPartialsIndex[nodeIndex]][nodeIndex], 0, partialsOut, 0, partialsOut.length);
    }

    /**
     * Allocates states for a node
     */
    public void createNodeStates(int nodeIndex) {

        this.states[nodeIndex] = new int[nrOfPatterns];
    }

    /**
     * Sets states for a node
     */
    @Override
    public void setNodeStates(int nodeIndex, int[] states) {

        if (this.states[nodeIndex] == null) {
            createNodeStates(nodeIndex);
        }
        System.arraycopy(states, 0, this.states[nodeIndex], 0, nrOfPatterns);
    }

    /**
     * Gets states for a node
     */
    @Override
    public void getNodeStates(int nodeIndex, int[] states) {
        System.arraycopy(this.states[nodeIndex], 0, states, 0, nrOfPatterns);
    }

    @Override
    public void setNodeMatrixForUpdate(int nodeIndex) {
        currentMatrixIndex[nodeIndex] = 1 - currentMatrixIndex[nodeIndex];

    }


    /**
     * Sets probability matrix for a node
     */
    @Override
    public void setNodeMatrix(int nodeIndex, int matrixIndex, double[] matrix) {
        System.arraycopy(matrix, 0, matrices[currentMatrixIndex[nodeIndex]][nodeIndex],
                matrixIndex * matrixSize, matrixSize);
    }

    public void setPaddedNodeMatrices(int nodeIndex, double[] matrix) {
        System.arraycopy(matrix, 0, matrices[currentMatrixIndex[nodeIndex]][nodeIndex],
                0, nrOfMatrices * matrixSize);
    }


    /**
     * Gets probability matrix for a node
     */
    @Override
    public void getNodeMatrix(int nodeIndex, int matrixIndex, double[] matrix) {
        System.arraycopy(matrices[currentMatrixIndex[nodeIndex]][nodeIndex],
                matrixIndex * matrixSize, matrix, 0, matrixSize);
    }

    @Override
    public void setNodePartialsForUpdate(int nodeIndex) {
        currentPartialsIndex[nodeIndex] = 1 - currentPartialsIndex[nodeIndex];
    }

    /**
     * Sets the currently updating node partials for node nodeIndex. This may
     * need to repeatedly copy the partials for the different category partitions
     */
    public void setCurrentNodePartials(int nodeIndex, double[] partials) {
        if (partials.length < partialsSize) {
            int k = 0;
            for (int i = 0; i < nrOfMatrices; i++) {
                System.arraycopy(partials, 0, this.partials[currentPartialsIndex[nodeIndex]][nodeIndex], k, partials.length);
                k += partials.length;
            }
        } else {
            System.arraycopy(partials, 0, this.partials[currentPartialsIndex[nodeIndex]][nodeIndex], 0, partials.length);
        }
    }

    /**
     * Calculates partial likelihoods at a node.
     *
     * @param nodeIndex1 the 'child 1' node
     * @param nodeIndex2 the 'child 2' node
     * @param nodeIndex3 the 'parent' node
     */
    @Override
    public void calculatePartials(int nodeIndex1, int nodeIndex2, int nodeIndex3) {
        if (states[nodeIndex1] != null) {
            if (states[nodeIndex2] != null) {
                calculateStatesStatesPruning(
                        states[nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        states[nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3]);
            } else {
                calculateStatesPartialsPruning(states[nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        partials[currentPartialsIndex[nodeIndex2]][nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3]);
            }
        } else {
            if (states[nodeIndex2] != null) {
                calculateStatesPartialsPruning(states[nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex1]][nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3]);
            } else {
                calculatePartialsPartialsPruning(partials[currentPartialsIndex[nodeIndex1]][nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        partials[currentPartialsIndex[nodeIndex2]][nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3]);
            }
        }

        if (useScaling) {
            scalePartials(nodeIndex3);
        }

//
//        int k =0;
//        for (int i = 0; i < patternCount; i++) {
//            double f = 0.0;
//
//            for (int j = 0; j < stateCount; j++) {
//                f += partials[currentPartialsIndices[nodeIndex3]][nodeIndex3][k];
//                k++;
//            }
//            if (f == 0.0) {
//                Logger.getLogger("error").severe("A partial likelihood (node index = " + nodeIndex3 + ", pattern = "+ i +") is zero for all states.");
//            }
//        }
    }

    /**
     * Calculates partial likelihoods at a node.
     *
     * @param nodeIndex1 the 'child 1' node
     * @param nodeIndex2 the 'child 2' node
     * @param nodeIndex3 the 'parent' node
     * @param matrixMap  a map of which matrix to use for each pattern (can be null if integrating over categories)
     */
    public void calculatePartials(int nodeIndex1, int nodeIndex2, int nodeIndex3, int[] matrixMap) {
        if (states[nodeIndex1] != null) {
            if (states[nodeIndex2] != null) {
                calculateStatesStatesPruning(
                        states[nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        states[nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3], matrixMap);
            } else {
                calculateStatesPartialsPruning(
                        states[nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        partials[currentPartialsIndex[nodeIndex2]][nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3], matrixMap);
            }
        } else {
            if (states[nodeIndex2] != null) {
                calculateStatesPartialsPruning(
                        states[nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex1]][nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3], matrixMap);
            } else {
                calculatePartialsPartialsPruning(
                        partials[currentPartialsIndex[nodeIndex1]][nodeIndex1], matrices[currentMatrixIndex[nodeIndex1]][nodeIndex1],
                        partials[currentPartialsIndex[nodeIndex2]][nodeIndex2], matrices[currentMatrixIndex[nodeIndex2]][nodeIndex2],
                        partials[currentPartialsIndex[nodeIndex3]][nodeIndex3], matrixMap);
            }
        }

        if (useScaling) {
            scalePartials(nodeIndex3);
        }
    }


    @Override
    public void integratePartials(int nodeIndex, double[] proportions, double[] outPartials) {
        calculateIntegratePartials(partials[currentPartialsIndex[nodeIndex]][nodeIndex], proportions, outPartials);
    }


    /**
     * Scale the partials at a given node. This uses a scaling suggested by Ziheng Yang in
     * Yang (2000) J. Mol. Evol. 51: 423-432
     * <p/>
     * This function looks over the partial likelihoods for each state at each pattern
     * and finds the largest. If this is less than the scalingThreshold (currently set
     * to 1E-40) then it rescales the partials for that pattern by dividing by this number
     * (i.e., normalizing to between 0, 1). It then stores the log of this scaling.
     * This is called for every internal node after the partials are calculated so provides
     * most of the performance hit. Ziheng suggests only doing this on a proportion of nodes
     * but this sounded like a headache to organize (and he doesn't use the threshold idea
     * which improves the performance quite a bit).
     *
     * @param nodeIndex
     */
    protected void scalePartials(int nodeIndex) {
//        int v = 0;
//    	double [] partials = m_fPartials[m_iCurrentPartialsIndices[nodeIndex]][nodeIndex];
//        for (int i = 0; i < m_nPatternCount; i++) {
//            for (int k = 0; k < m_nMatrixCount; k++) {
//                for (int j = 0; j < m_nStateCount; j++) {
//                	partials[v] *= SCALE;
//                	v++;
//                }
//            }
//        }
        int u = 0;

        for (int i = 0; i < nrOfPatterns; i++) {

            double scaleFactor = 0.0;
            int v = u;
            for (int k = 0; k < nrOfMatrices; k++) {
                for (int j = 0; j < nrOfStates; j++) {
                    if (partials[currentPartialsIndex[nodeIndex]][nodeIndex][v] > scaleFactor) {
                        scaleFactor = partials[currentPartialsIndex[nodeIndex]][nodeIndex][v];
                    }
                    v++;
                }
                v += (nrOfPatterns - 1) * nrOfStates;
            }

            if (scaleFactor < scalingThreshold) {

                v = u;
                for (int k = 0; k < nrOfMatrices; k++) {
                    for (int j = 0; j < nrOfStates; j++) {
                        partials[currentPartialsIndex[nodeIndex]][nodeIndex][v] /= scaleFactor;
                        v++;
                    }
                    v += (nrOfPatterns - 1) * nrOfStates;
                }
                scalingFactors[currentPartialsIndex[nodeIndex]][nodeIndex][i] = Math.log(scaleFactor);

            } else {
                scalingFactors[currentPartialsIndex[nodeIndex]][nodeIndex][i] = 0.0;
            }
            u += nrOfStates;


        }
    }

    /**
     * This function returns the scaling factor for that pattern by summing over
     * the log scalings used at each node. If scaling is off then this just returns
     * a 0.
     *
     * @return the log scaling factor
     */





    @Override
    public double getLogScalingFactor(int patternIndex_) {
//    	if (m_bUseScaling) {
//    		return -(m_nNodeCount/2) * Math.log(SCALE);
//    	} else {
//    		return 0;
//    	}
        double logScalingFactor = 0.0;
        if (useScaling) {
            for (int i = 0; i < nrOfNodes; i++) {
                logScalingFactor += scalingFactors[currentPartialsIndex[i]][i][patternIndex_];
            }
        }
        return logScalingFactor;
    }

    /**
     * Gets the partials for a particular node.
     *
     * @param nodeIndex   the node
     * @param outPartials an array into which the partials will go
     */
    public void getPartials(int nodeIndex, double[] outPartials) {
        double[] partials1 = partials[currentPartialsIndex[nodeIndex]][nodeIndex];

        System.arraycopy(partials1, 0, outPartials, 0, partialsSize);
    }

    /**
     * Store current state
     */
    @Override
    public void restore() {
        // Rather than copying the stored stuff back, just swap the pointers...
        int[] tmp1 = currentMatrixIndex;
        currentMatrixIndex = storedMatrixIndex;
        storedMatrixIndex = tmp1;

        int[] tmp2 = currentPartialsIndex;
        currentPartialsIndex = storedPartialsIndex;
        storedPartialsIndex = tmp2;
    }

    @Override
    public void unstore() {
        System.arraycopy(storedMatrixIndex, 0, currentMatrixIndex, 0, nrOfNodes);
        System.arraycopy(storedPartialsIndex, 0, currentPartialsIndex, 0, nrOfNodes);
    }

    /**
     * Restore the stored state
     */
    @Override
    public void store() {
        System.arraycopy(currentMatrixIndex, 0, storedMatrixIndex, 0, nrOfNodes);
        System.arraycopy(currentPartialsIndex, 0, storedPartialsIndex, 0, nrOfNodes);
    }


//	@Override
//    public void calcRootPsuedoRootPartials(double[] frequencies, int nodeIndex, double [] pseudoPartials) {
//		int u = 0;
//		double [] inPartials = m_fPartials[m_iCurrentPartials[nodeIndex]][nodeIndex];
//		for (int k = 0; k < m_nPatterns; k++) {
//			for (int l = 0; l < m_nMatrices; l++) {
//				for (int i = 0; i < m_nStates; i++) {
//					pseudoPartials[u] = inPartials[u] * frequencies[i];
//					u++;
//				}
//			}
//		}
//    }
//	@Override
//    public void calcNodePsuedoRootPartials(double[] inPseudoPartials, int nodeIndex, double [] outPseudoPartials) {
//		double [] partials = m_fPartials[m_iCurrentPartials[nodeIndex]][nodeIndex];
//		double [] oldPartials = m_fPartials[m_iStoredPartials[nodeIndex]][nodeIndex];
//		int maxK = m_nPatterns * m_nMatrices * m_nStates;
//		for (int k = 0; k < maxK; k++) {
//			outPseudoPartials[k] = inPseudoPartials[k] * partials[k] / oldPartials[k];
//		}
//	}
//
//	@Override
//    public void calcPsuedoRootPartials(double [] parentPseudoPartials, int nodeIndex, double [] pseudoPartials) {
//		int v = 0;
//		int u = 0;
//		double [] matrices = m_fMatrices[m_iCurrentMatrices[nodeIndex]][nodeIndex];
//		for (int k = 0; k < m_nPatterns; k++) {
//			for (int l = 0; l < m_nMatrices; l++) {
//				for (int i = 0; i < m_nStates; i++) {
//					int w = 0;
//					double sum = 0;
//					for (int j = 0; j < m_nStates; j++) {
//					      sum += parentPseudoPartials[u+j] * matrices[w + i];
//					      w+=m_nStates;
//					}
//					pseudoPartials[v] = sum;
//					v++;
////					int w = l * m_nMatrixSize;
////					double sum = 0;
////					for (int j = 0; j < m_nStates; j++) {
////					      sum += parentPseudoPartials[u+j] * matrices[w+j];
////					}
////					pseudoPartials[v] = sum;
////					v++;
//				}
//				u += m_nStates;
//			}
//		}
//    }
//
//
//    @Override
//    void integratePartialsP(double [] inPartials, double [] proportions, double [] m_fRootPartials) {
//		int maxK = m_nPatterns * m_nStates;
//		for (int k = 0; k < maxK; k++) {
//			m_fRootPartials[k] = inPartials[k] * proportions[0];
//		}
//
//		for (int l = 1; l < m_nMatrices; l++) {
//			int n = maxK * l;
//			for (int k = 0; k < maxK; k++) {
//				m_fRootPartials[k] += inPartials[n+k] * proportions[l];
//			}
//		}
//    } // integratePartials
//
//	/**
//	 * Calculates pattern log likelihoods at a node.
//	 * @param partials the partials used to calculate the likelihoods
//	 * @param frequencies an array of state frequencies
//	 * @param outLogLikelihoods an array into which the likelihoods will go
//	 */
//    @Override
//	public void calculateLogLikelihoodsP(double[] partials,double[] outLogLikelihoods)
//	{
//        int v = 0;
//		for (int k = 0; k < m_nPatterns; k++) {
//            double sum = 0.0;
//			for (int i = 0; i < m_nStates; i++) {
//				sum += partials[v];
//				v++;
//			}
//            outLogLikelihoods[k] = Math.log(sum) + getLogScalingFactor(k);
//		}
//	}
//
//
//	//    @Override
////    LikelihoodCore feelsGood() {return null;}

    @Override
    public boolean getUseScaling() {
        return useScaling;
    }

} // class BeerLikelihoodCore





























package beast.evolution.substitutionmodel;
import beast.evolution.datatype.DataType;
import beast.evolution.datatype.IntegerData;
import beast.evolution.tree.Node;

import java.util.Arrays;
import jdk.incubator.vector.*;

import beast.core.Input;
import beast.core.Input.Validate;
import beast.core.parameter.RealParameter;


public lass BD extends SubstitutionModel.Base {
	public Input<RealParameter> nstate = new Input<RealParameter>("nstate", "same as what in BD model", Validate.REQUIRED);
	protected static int nrOfStates;
	protected static double[] binom_array;
	protected static double[] db_resultsi;
	protected double[] db_resultsj;
	protected VectorSpecies<Float> a = FloatVector.SPECIES_256;
	protected VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
	protected Vector<Double> v0 = SPECIES.broadcast (0);
	protected Vector<Double> v1 = SPECIES.broadcast (1);
	protected int upperBound = SPECIES.loopBound(nrOfStates);

	public BD() {
		// this is added to avoid a parsing error inherited from superclass because frequencies are not provided.
		frequenciesInput.setRule(Validate.OPTIONAL);
		try {
			// this call will be made twice when constructed from XML
			// but this ensures that the object is validly constructed for testing purposes.
			//System.out.println("Class Constructor");
			//System.out.println(nstate.get().getValue());
			//initAndValidate();
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("initAndValidate() call failed when constructing BD()");
		}
	}

	@Override
	public void initAndValidate() {
		int i, j, index;
		super.initAndValidate();
		//System.out.println("BD");
		//System.out.println(nstate.get().getValue());
		if (nstate.get() == null) {
			throw new IllegalArgumentException("number of states to consider is required");
		}
		nrOfStates = (int)Math.round(nstate.get().getValue());
		//nrOfStates = 30;
		binom_array = new double[nrOfStates * nrOfStates];
		db_resultsi = new double[nrOfStates * nrOfStates];
		db_resultsj = new double[nrOfStates * nrOfStates];
		for (i = 0; i < nrOfStates ; ++i) {
			for (j = 0; j < nrOfStates ; ++j) {
				index = i * nrOfStates + j;
				binom_array[index] = binomialCoeff(i,j);
				db_resultsi[index] = i;
				db_resultsj[index] = j;
			}
		}
		//trMatrix = new double[(nrOfStates - 1) * (nrOfStates - 1)];
	}


	//public static final int nstates = 30;
	@Override
	public double[] getFrequencies() {
		return null;
	}

	public int getStateCount() {
		return nrOfStates;
	}
	//protected int nrOfStates = 30;



	public static double binomi(int n, int k) {
		return binom_array[n*nrOfStates+k];
	}
	public static double bd_prob(int child, int ancestor, double bd_rate, double distance) {
		double p = 0;
		int j;
		double p_j;
		int range = Math.min(child, ancestor) + 1;
		if (distance <=1) {
			for (j = 1; j < range; ++j ){
				p_j = binomi(ancestor, j) * binomi(child - 1, j - 1) * Math.pow(bd_rate * distance, child + ancestor -2*j);
				p += p_j;
			}
			p = p * Math.pow(bd_rate / (1 + distance * bd_rate), child + ancestor);
			//if (Math.pow(bd_rate / (1 + distance * bd_rate), child + ancestor) == 0) {
			//p = 0;
			//}
		}
		else {
			for (j = 1; j < range; ++j ){
				p_j = binomi(ancestor, j) * binomi(child - 1, j - 1) * Math.pow(bd_rate * distance, -2 * j);
				p += p_j;
			}
			p = p * Math.pow(bd_rate*distance / (1 + distance * bd_rate), child + ancestor);
			if (Math.pow(bd_rate*distance / (1 + distance * bd_rate), child + ancestor) == 0) {
				//p = 0;
			}
		}

		return p;
	}

	protected boolean checkTransitionMatrix(double[] matrix) {
		double sum = 0;
		int i, j;
		int index;
		for (i = 0; i < nrOfStates ; ++i) {
			for (j = 0; j < nrOfStates ; ++j) {
				index = i * nrOfStates + j;
				sum = sum + matrix[index];
			}
			if (sum > 1.01 |sum < 0.95) {
				//System.out.println("current index:" + i);
				//System.out.println(sum);
				return true;
			}
			sum = 0;
		}
		return true;

	}

	public double getProbability(int i, int j,double distance, double bd_rate){
		if(i == 0){
			if (j == 0) {
				return 1.0;
			}
			else {
				return 0.0;
			}
		}
		else if(j == 0){
			return Math.pow((bd_rate * distance) / (1 + bd_rate * distance), i);
		}
		else if (i == 1){
			return Math.pow(distance, j - 1) / Math.pow((1 + distance), j + 1);
		}
		else{
			return bd_prob(i, j, distance, bd_rate);
		}
	}

	@Override
	public void getTransitionProbabilities(Node node, double startTime, double endTime, double rate, double[] matrix) {
		// TODO Auto-generated method stub
		//assume birth rate = death rate = 1
		//System.out.println("TRANSITIONPROB");
		double bd_rate = 1;
		int index;
		int i = 0, j = 0;
		double distance = (startTime - endTime) * rate;
		for (; i < nrOfStates; i ++) {
			j = 0;
			for (; j < upperBound ; j += SPECIES.length()) {
				index = i * nrOfStates + j;
				DoubleVector vm = DoubleVector.fromArray(SPECIES, matrix, index);
				DoubleVector vprei = DoubleVector.fromArray(SPECIES, db_resultsi, index);
				DoubleVector vprej = DoubleVector.fromArray(SPECIES, db_resultsj, index);
				System.out.println(matrix[index]);
				System.out.println(db_resultsj[index]);
				vm=vm.add(1,vprei.eq(v0).and(vprej.eq(v0)))
						.add(0,vprei.eq(v0).and(vprej.eq(v0).not()))
						.add(Math.pow((bd_rate * distance) / (1 + bd_rate * distance), i),vprei.eq(v0).not().and(vprej.eq(v0)))
						.add((Math.pow(distance, j - 1) / Math.pow((1 + distance), j + 1)),vprei.eq(v1).and(vprej.eq(v0).not()))
						.add(bd_prob(i, j, distance, bd_rate),vprei.eq(v0).not().and(vprei.eq(v1).not()).and(vprej.eq(v0).not()));
				vm.intoArray(matrix, index);
			}
		}

		for (; i < nrOfStates ; i++) {
			for (; j < nrOfStates ; j++) {
				index = i * nrOfStates + j;
				matrix[index] = getProbability(i,j,distance,bd_rate);
			}
		}

	}
	public int binomialCoeff(int n, int k) {
		int res = 1;

		// Since C(n, k) = C(n, n-k)
		if (k > n - k)
			k = n - k;

		// Calculate value of
		// [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
		for (int i = 0; i < k; ++i) {
			res *= (n - i);
			res /= (i + 1);
		}

		return res;
	}

	@Override
	public EigenDecomposition getEigenDecomposition(Node node) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean canHandleDataType(DataType dataType) {
		// TODO Auto-generated method stub
		return dataType instanceof IntegerData;
	}

}c




        double p = 0;
		int j;
		double p_j;
		int range = Math.min(child, ancestor) + 1;

        .add(1, vprei[index], bd_rate,distance),vprei.eq(0).not().and(vprei.eq(1).not()).and(vprej.eq(0).not()))
        
        .add(1, vprei[index], bd_rate,distance),vprei.eq(0).not().and(vprei.eq(1).not()).and(vprej.eq(0).not()))


		if (distance <= 1) {
			for (j = 1; j < range; ++j) {
				p_j = binomi(ancestor, j) * binomi(child - 1, j - 1) * Math.pow(bd_rate * distance, child + ancestor - 2 * j);
				p += p_j;
			}
			p = p * Math.pow(bd_rate / (1 + distance * bd_rate), child + ancestor);
			//if (Math.pow(bd_rate / (1 + distance * bd_rate), child + ancestor) == 0) {
			//p = 0;
			//}
		} else {
			for (j = 1; j < range; ++j) {
				p_j = binomi(ancestor, j) * binomi(child - 1, j - 1) * Math.pow(bd_rate * distance, -2 * j);
				p += p_j;
			}
			p = p * Math.pow(bd_rate * distance / (1 + distance * bd_rate), child + ancestor);
			if (Math.pow(bd_rate * distance / (1 + distance * bd_rate), child + ancestor) == 0) {
				//p = 0;
			}
		}

		return p;
