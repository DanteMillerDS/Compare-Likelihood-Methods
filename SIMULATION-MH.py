#IMPORTS
from ast import Return
import numpy as np
from scipy.stats import lognorm
import dendropy
from Bio import AlignIO
from io import StringIO
from collections import Counter
import math
from biotite.sequence.phylo import upgma
from Bio import Phylo
import emcee
from matplotlib import pyplot as plt
np.random.seed(123)



class TreeSequenceGeneration():

    '''
    This class is used to generate a phylogenetic tree,
    generate a sequence based on the phylogenetic tree,
    generate phylgoenetic tree based on sequence generated,
    performs metropolis hastings sampling using joint density
    likelihood and non joint density likelihood functions
    '''

    def __init__(self,SHAPE)->None: 
        self.SHAPE=SHAPE # PASS A SHAPE WHEN CREATING OBJECT
        # VARIABLES BELOW ARE USED IN THE OTHER FUNCTIONS FOR THE CLASS
        self.THETA=""
        self.TREE=""
        self.SEQUENCE=""
        self.SEQSEQ=""
        self.TAXA=""
        self.POPULATIONSIZE=""
        self.COALESCENTTTIME=""
        self.GENERATED_TREE=""
        self.TIMES=""
        self.BRANCHLENGTH=dict()
        self.TREESTRING=""
        self.STRINGUSEDDICTIONARY=dict()
        self.JFINAL=[]
        self.NJFINAL=[]
        self.ALLNEWICKSTRING=[]
        self.SPECIESDICTIONARY=dict()
        self.FINALTIMEARRAY=[]
        self.MUTATIONDICTIONARY=dict()

    '''
    This function generates a random theta value
    based on the shape parameter.
    '''

    def GenerateTheta(self):
        self.THETA=lognorm.rvs(self.SHAPE, size=1) # GENERATE THETA FROM LOG NORM DISTRIBUTION SHAPE REPRESENTS THE MAX VALUE THAT CAN BE DRAWN
        return self.THETA
    
    '''
    This function simulates a tree with a specific 
    number of taxa and on a pure kingman tree process using a 
    population size based on the theta function generated from
    the GenerateTheta function.
    '''

    def SimulateTree(self):
        self.TAXA=dendropy.TaxonNamespace(["z1", "z2", "z3","z4"]) # AMOUNT OF TAXAS/SPECIES
        self.POPULATIONSIZE=2/self.THETA # POPULATION SIZE FORMULA
        self.TREE=dendropy.simulate.treesim.pure_kingman_tree(taxon_namespace=self.TAXA,pop_size=self.POPULATIONSIZE) # GENERATE A TREE BASED ON POPULATION SIZE AND TAXA/SPECIES
        print()
        print("Tree Model Simulation: ")
        print()
        print("Theta: "+str(self.THETA[0]))
        print("Printed and Visualized Tree Model: " )
        print(str(self.TREE))
        self.TREE.print_plot()
        print("--------------------------------------------------------------------------------------------------")
        print()
    
    '''
    This function generates a sequence of a specific 
    using the phylogenetic tree generated from the SimulateTree 
    function.
    '''

    def SimulateSequence(self):
        self.SEQUENCE=dendropy.model.discrete.hky85_chars(1000,tree_model=self.TREE) # SIMULATE SEQEUNCES OF THE TREE UP TO 1000 CHARACTERS
        self.SEQUENCE=self.SEQUENCE.as_string(schema="phylip") # SHOW SEQUENCE USING THE PHYLIP SCHEME
        print("Sequence Derived from Tree Simulation")
        print()
        print(self.SEQUENCE)
        print("--------------------------------------------------------------------------------------------------")
        print()
    
    '''
    This function generates a similarity score
    based on two sequences provided. This function
    will have to be altered for insertions and deletions.
    I believe it can be altered to take the alignment score. What should be mentioned
    is that the sequences are strings with a counter function applied to it thus you will have to
    alter the GenerateTreeForSequence function.
    '''

    def ComputeSimilarity(self,SEQUENCEONE,SEQUENCETWO):
        # NEEDS TO WORK FOR INSERTIONS AND DELETIONS SO TAKE THE ALIGNMENT OF THE SETS AND CALCULATE SCORE?
        TERMS=set(SEQUENCEONE).union(SEQUENCETWO) # TAKES THE UNION OF THE TWO SETS
        DOTPRODUCT=sum(SEQUENCEONE.get(k,0)*SEQUENCETWO.get(k,0)for k in TERMS) # MULTIPLY OCCURENCES OF EACH NUCLEOTIDE IN SEQUENCEONE BY SEQUENCETWO THEN TAKE SUM
        MAGA=math.sqrt(sum(SEQUENCEONE.get(k,0)**2 for k in TERMS)) # TAKE THE OCCURENCE OF EACH NUCLEOTIDE SQUARED THEN TAKE SUM SEQUENCEONE
        MAGB=math.sqrt(sum(SEQUENCETWO.get(k,0)**2 for k in TERMS)) # TAKE THE OCCURENCE OF EACH NUCLEOTIDE SQUARED THEN TAKE SUM SEQUENCETWO
        return DOTPRODUCT/(MAGA*MAGB) # RETURNS THE DOT PRODUCT SCORE

    '''
    This function generates an alignment from the sequences. It then 
    computes the distance matrix which the generate phylogenetic tree method 
    upgma is applied to it. It then produces a string which I swap the numbers with the respecitive species.
    I then used that string to generate a phylogenetic tree. The similarity scores used in the distance matrix
    will have to be altered to include isnertions and deletions which will require the ComputeSimilarity function
    to be altered too. I used upgma as the phylogenetic tree generation method but I think it can be swapped for
    a more simplier method such as neighbor joining (neighbor_joining).
    '''
    
    def GenerateTreeForSequence(self):
        ALN=AlignIO.read(StringIO(self.SEQUENCE),'phylip') # INPUTS THE SEQUENCES AS AN ALIGNMENT
        DISTANCEMATRIX=np.zeros([len(ALN), len(ALN)]) # INTIATE DISTANCE MATRIX
        for I,SPECIESONE in enumerate(ALN): # ITERATE THROUGH ALN
            SEQUENCEONE=SPECIESONE.id.split(" ", 1)[1][1:] # GRABBING SEQUENCEONE
            for J,SPECIESTWO in enumerate(ALN): # ITERATE THROUGH ALN
                SEQUENCETWO=SPECIESTWO.id.split(" ", 1)[1][1:] # GRABBING SEQUENCETWO
                DISTANCEMATRIX[I,J]=self.ComputeSimilarity(Counter(SEQUENCEONE), Counter(SEQUENCETWO)) # PROVIDES A DICTIONARY OF SEQUENCEONE AND SEQUENCETWO NUCLEOTIDES AND OCCURENCES TO A COSINESIMILARLITY FUNCTION
        SPECIESNAMES = [SP.id.split(" ",1)[0] for SP in ALN] # FINDS ALL THE SPECIES/TAXA NAMES
        UPDATEDDISTANCEMATRIX=list(str(upgma(DISTANCEMATRIX))) # CALLS UPGMA ON THE DISTANCE MATRIX WHICH RETURNS A STR WHICH IS THEN TURNED INTO A LIST
        for i in range(len(UPDATEDDISTANCEMATRIX)): # ITERATE THROUGH STRING
            if(str(upgma(DISTANCEMATRIX))[i].isdigit() and str(upgma(DISTANCEMATRIX))[i+1]==":"): # IF STATEMENT CHECKING FOR NUMBERS THAT REPRESENT THE SPECIES 
                UPDATEDDISTANCEMATRIX[i]=SPECIESNAMES[int(UPDATEDDISTANCEMATRIX[i])] # REPLACES NUMBERS FOR SPECIES WITH A SPECIFIC TAXA LABEL
        UPDATEDDISTANCEMATRIX=''.join(UPDATEDDISTANCEMATRIX) # JOINS THE CHARACTERS AS ONE STRING
        HANDLE=StringIO(UPDATEDDISTANCEMATRIX) # USES STRING AS A FILE OBJECT
        self.GENERATED_TREE=Phylo.read(HANDLE, "newick") # READS THE NEWICK STRING AS A TREE
        self.GENERATED_TREE.rooted=True # SET THE TREE TO BE ROOTED
        self.GENERATED_TREE.ladderize() # REORGANIZES TREE
        print("Generated Tree from Sequences")
        print()   
        Phylo.draw_ascii(self.GENERATED_TREE) # DRAWS TREE
        print("--------------------------------------------------------------------------------------------------")
        print()

    '''
    This function is returning the branch lengths of the 
    tree for the speices and last common ancestor nodes
    '''

    def ReturnBranchLengths(self):
        NUMBER=1 #INTIAL
        for NODE in self.GENERATED_TREE.find_clades(branch_length=True,order="preorder"): # ITERATE THROUGH TREE
            if(str(NODE)=="Clade"): # IF NODE IS A CLADE
                self.BRANCHLENGTH[str(NODE)+str(NUMBER)]=NODE.branch_length # THEN ADD THE BRANCH LENGTH FOR THE CLADE INTO DICTIONARY
                NUMBER=NUMBER+1 # INCREASE NUMBER BY ONE
            else:
                self.BRANCHLENGTH[str(NODE)]=NODE.branch_length # THEN ADD BRANCH LENGTH FOR THE SPECIES/TAXA INTO DICTIONARY
        print("Branch Lengths for Generated Tree")
        print()   
        print(self.BRANCHLENGTH)
        print("--------------------------------------------------------------------------------------------------")
        print()

    '''
    This function return the result of the beta function for
    a provided theta and time value.
    '''

    def BETA(self,THETA,TIME):
        return THETA/(TIME-1) # BETA FUNCTION

    '''
    This function return the result of the coalescent time function for
    a provided theta and time value.
    '''

    def COALESCENTTIME(self,THETA,TIME):
        return ((THETA)/(TIME*(TIME-1))) # COALESCENT TIME FUNCTION

    '''
    This function generates the maximum time of the tree based on times
    from the taken from the BRANCHLENGTH dictionary and that the first time is t2 or 2.
    '''

    def ReturnTimes(self):
        TIMES=list(self.BRANCHLENGTH.keys()) # CREATES LIST OF CREATES
        COPY=[] # CREATES COPY ARRAY
        for TIME in TIMES: # ITERATE THROUGH BRANCHES
            if "Clade" in str(TIME):
                COPY.append(TIME) # APPENDS BRANCHES THAT ARE NOT CLADE TO THE COPY
        LENGTH=len(COPY)+2 # RETURNS MAXIMUM TIME
        return LENGTH

    '''
    This function generates a mutation dictionary by calculating the 
    number of mutations on a species branch using sequence length being 1000
    and the mutation function.
    '''

    def FindMutations(self):
        MUTATIONDICTIONARY=dict() # CREATES MUTATION DICTIONARY
        SEQUENCELENGTH=1000 # SEQUENCE LENGTH
        for TAXON in self.BRANCHLENGTH: # ITERATES THROUGH BRANCHES
            if "Clade" not in TAXON: 
                MUTATION=int(round((SEQUENCELENGTH*self.BRANCHLENGTH[TAXON])/100)) # FINDS AMOUNT OF MUTATIONS ON EACH BRANCH
                MUTATIONDICTIONARY[TAXON]=MUTATION # SETS THE BRANCH VALUE IN DICTIONARY TO THE MUTATION VALUE
        return MUTATIONDICTIONARY

    '''
    ADD COMMENTS
    '''

    def FindSpeciesTimes(self,BRANCHLENGTH):
         MUTATIONS=dict()
         TIMES=2
         BRANCHLIST=list(BRANCHLENGTH.keys())
         BRANCHLIST=list(reversed(BRANCHLIST))
         [BRANCHLIST.remove(CLADE) for CLADE in BRANCHLIST[:] if "Clade" not in CLADE]
         for CLADE in BRANCHLIST:
            ONLYTWO=0
            NEXTTWO=False
            for SPECIES in BRANCHLENGTH:
                if CLADE == SPECIES:
                    NEXTTWO=True
                elif NEXTTWO==True and ONLYTWO<2:
                    if "Clade" not in SPECIES:
                        MUTATIONS[SPECIES]=TIMES
                    ONLYTWO=ONLYTWO+1
            TIMES=TIMES+1
         [MUTATIONS.update({CLADE:TIMES}) for CLADE in BRANCHLENGTH if "Clade" not in CLADE and CLADE not in list(MUTATIONS.keys())]
         DUPLICATEARRAY=[]
         for MUTATION in list(MUTATIONS.keys()):
            K = [k for k,v in MUTATIONS.items() if v == MUTATIONS[MUTATION]]
            if K not in DUPLICATEARRAY and len(K) >=2:
                DUPLICATEARRAY.append(K)
         for DUPLICATEINDEX in range(len(DUPLICATEARRAY)):
            for DUPLICATEINDEXONE in range(DUPLICATEINDEX,len(DUPLICATEARRAY)):
                if(MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]]>MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][0]] and 
                BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEX][0]]<BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEXONE][0]] or
                MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]]<MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][0]] and 
                BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEX][0]]>BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEXONE][0]]):
                   EXCHANGEONE=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]]
                   EXCHANGETWO=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][1]]
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]]=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][1]]
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][1]]=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][1]]
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][0]]=EXCHANGEONE
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][1]]=EXCHANGETWO
         return MUTATIONS

    '''
    ADD COMMENTS
    '''
    
    def Prior(self,THETA):
        if THETA < 2 and THETA > 0:
            return 1
        return .5

    '''
    ADD COMMENTS
    '''

    def LLNJ(self,THETA):
        LP = self.Prior(THETA)
        FINAL_VALUE=1
        for TIME in range(2,self.TIMES+1):
            BETAVALUE=self.BETA(THETA[0],TIME)
            FINAL_VALUE=FINAL_VALUE*(1/(BETAVALUE+1))*((BETAVALUE/(BETAVALUE+1))**self.MUTATIONDICTIONARY[TIME])
        return LP + FINAL_VALUE

    '''
    ADD COMMENTS
    '''

    def LLJ(self,THETA):
        LP = self.Prior(THETA)
        FINALVALUE=1
        for TIME in range(2,self.TIMES+1):
            #POWER=(1*TIME*self.COALESCENTTIME(THETA[0],TIME))
            FINALVALUE=FINALVALUE*((self.MUTATIONDICTIONARY[TIME][0])*(self.COALESCENTTIME(THETA[0],TIME)))
        return LP + FINALVALUE
    
    '''
    ADD COMMENTS
    '''

    def MHastings(self,DICTIONARY):
            print("New Topology")
            print(DICTIONARY)
            self.MUTATIONDICTIONARY=DICTIONARY
            NWALKER=2
            NDIM=1
            POSITIONJOINT = [lognorm.rvs(self.SHAPE, size=1) for i in range(NWALKER)]
            POSITIONNONJOINT = POSITIONJOINT
            JOINTSAMPLER = emcee.EnsembleSampler(NWALKER,NDIM,self.LLNJ,args=())
            JOINTSAMPLER.run_mcmc(POSITIONJOINT,1)
            print("Joint Sampler")
            print(JOINTSAMPLER.run_mcmc(POSITIONJOINT,1)[1])
            NONJOINTSAMPLER = emcee.EnsembleSampler(NWALKER,NDIM,self.LLJ,args=())
            print("Non Joint Sampler")
            print(NONJOINTSAMPLER.run_mcmc(POSITIONNONJOINT,1)[1])
    
    '''
    ADD COMMENTS
    '''
    
    def callBackMutation(self,TIME,CURRENTTIME,SPECIE,PASSDICTIONARY,CARRY,MUTATION):
        if CURRENTTIME-1<TIME: # CURRENT TIME MUST BE LESS THAN THE MAXIMUM TIME
            for TIMEPLACEMENT in range(1, CARRY+2): 
                PASSDICTIONARY[CURRENTTIME] = TIMEPLACEMENT-1
                CARRYOVER=CARRY-PASSDICTIONARY[CURRENTTIME]
                self.callBackMutation(TIME,CURRENTTIME+1,SPECIE,PASSDICTIONARY,CARRYOVER,MUTATION) # CALLS THE MUTATION FUNCTION FOR THE NEXT TIME
        elif CURRENTTIME-1==TIME:
            PASSDICTIONARY[CURRENTTIME] = CARRY
            self.SPECIESDICTIONARY[SPECIE].append(PASSDICTIONARY.copy())
    
    '''
    ADD COMMENTS
    '''

    def callBacks(self,CALLBACKS,MUTATIONS,SPECIETIMES):
       if CALLBACKS != []: # IF THE SPECIES ARRAY IS NOT EMPTY
            SPECIE=CALLBACKS[len(CALLBACKS)-1] # GRABS THE SPECIE AT THE LOWEST TIME
            CALLBACKS.remove(SPECIE) # REMOVE ITS FROM THE SPECIES ARRAY
            MUTATION=MUTATIONS[SPECIE] # GRABS THE SPECIES ESTIMATED TIME VALUE
            TIME=SPECIETIMES[SPECIE] # GRABS THE MAXIMUM TIME FOR THE SPECIES
            CURRENTTIME=2 # DEFINES THE CURRENT TIME TO BE 2
            self.SPECIESDICTIONARY[SPECIE]=[] # ADDS AN ARRAY IN THE SPECIES DICTIONARY FOR THE SELECTED SPECIE
            PASSDICTIONARY=dict() # DEFINES A NEW DICTIONARY
            for FIRSTTIME in range(1,int(MUTATION)+2): # ITERATES THROUGH THE MUTATION PLACEMENT FOR THE FIRST TIME
                PASSDICTIONARY[CURRENTTIME]=FIRSTTIME-1 # PLACES THE NUMBER OF MUTATIONS IN THE DICTIONARY FOR THE SPECIES
                CARRYOVER=MUTATION-(PASSDICTIONARY[CURRENTTIME])
                self.callBackMutation(TIME,CURRENTTIME+1,SPECIE,PASSDICTIONARY,CARRYOVER,MUTATION)
            if CALLBACKS != []: # IF THE SPECIES ARRAY IS NOT EMPTY EMPTY
                return self.callBacks(CALLBACKS,MUTATIONS,SPECIETIMES) # RECALLS ITSELF

    '''
    ADD COMMENTS
    '''

    def IterateThroughIntervalDictionary(self,INTERVALDICTIONARY,KEYS,INDEX,TIMEDICTIONARY):
        if INDEX < len(KEYS):
            ARRAYS=INTERVALDICTIONARY[KEYS[INDEX]]
            for ARRAY in ARRAYS:
                COPY=TIMEDICTIONARY.copy()
                for MINIARRAY in list(reversed(ARRAY)):
                    #if INDEX + 1 < len(KEYS)-1:
                        VALUE=sum(COPY[MINIARRAY])
                        COPY[MINIARRAY] = [VALUE + ARRAY[MINIARRAY]]#[ARRAY[MINIARRAY]]
                self.IterateThroughIntervalDictionary(INTERVALDICTIONARY,KEYS,INDEX+1,COPY) 
        else:
            #print(sum(list(self.MUTATIONS.values())))
            #print(TIMEDICTIONARY)
            #print(self.MUTATIONS)
            SUMTIMEMUTATIONS=sum(TIMEDICTIONARY[MUTATION][0] for MUTATION in TIMEDICTIONARY)
            #print(SUMTIMEMUTATIONS)
            #print(sum(self.MUTATIONS.values()))
            if TIMEDICTIONARY not in self.FINALTIMEARRAY and SUMTIMEMUTATIONS==sum(self.MUTATIONS.values()):
                self.FINALTIMEARRAY.append(TIMEDICTIONARY.copy())
  
    '''
    ADD COMMENTS
    '''  

    def MHastingsLoop(self):
        self.TIMES=self.ReturnTimes()
        self.MUTATIONS=self.FindMutations()
        self.SPECIETIMES=self.FindSpeciesTimes(self.BRANCHLENGTH)
        CALLBACKS=[[]]
        for TAXON in self.BRANCHLENGTH:
            if "Clade" not in TAXON:
                CALLBACKS[0].append(TAXON)
        CALLBACKS.append(CALLBACKS[0][2:4]+CALLBACKS[0][0:2]) # NEED TO DESIGN A METHOD THAT GENERATES THESE COMBINATIONS
        for CALLBACK in CALLBACKS:
            self.SPECIESDICTIONARY=dict()
            self.callBacks(CALLBACK,self.MUTATIONS,self.SPECIETIMES)
            KEYS=list(self.SPECIESDICTIONARY.keys())
            TIMEDICTIONARY=dict.fromkeys(range(2,self.TIMES+2), [0])
            self.IterateThroughIntervalDictionary(self.SPECIESDICTIONARY,KEYS,0,TIMEDICTIONARY)
            #print(self.FINALTIMEARRAY)
            for DICTIONARY in self.FINALTIMEARRAY:
                DICTIONARY[self.TIMES] = [DICTIONARY[self.TIMES][0] + DICTIONARY[self.TIMES+1][0]]
                DICTIONARY[self.TIMES+1] = [0]
            self.FINALTIMEARRAY = [i for n, i in enumerate(self.FINALTIMEARRAY) if i not in self.FINALTIMEARRAY[n + 1:]]
            for DICTIONARY in self.FINALTIMEARRAY:
               self.MHastings(DICTIONARY)

'''
Calls the TreeSequenceGeneration class and runs 
the GenerateTheta, SimulateTree, SimulateSequence,
GenerateTreeForSequence, ReturnBranchLengths, and
MHastingsLoop functions.
'''

if __name__ == "__main__":
    TSG=TreeSequenceGeneration(1)
    TSG.GenerateTheta()
    TSG.SimulateTree()
    TSG.SimulateSequence()
    TSG.GenerateTreeForSequence()
    TSG.ReturnBranchLengths()
    TSG.MHastingsLoop()






