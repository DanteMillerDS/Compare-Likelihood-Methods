#IMPORTS
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
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
np.random.seed(123)
import seaborn as sns
import math
from mpmath import mp


class TreeSequenceGeneration():

    '''
    This class is used to generate a phylogenetic tree,
    generate a sequence based on the phylogenetic tree,
    generate phylgoenetic tree based on sequence generated,
    performs metropolis hastings sampling using joint density
    likelihood and non joint density likelihood functions
    '''

    def __init__(self,SHAPE,WALKER)->None: 
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
        self.MHASTINGPROBS=dict()
        self.TREEPROBS=dict()
        self.WALKERS=WALKER
        self.SEQUENCELENGTH=1000
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
        self.SEQUENCE=dendropy.model.discrete.hky85_chars(self.SEQUENCELENGTH,tree_model=self.TREE) # SIMULATE SEQEUNCES OF THE TREE UP TO 1000 CHARACTERS
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

    # DOES NOT WORK FOR INSERTIONS AND DELETIONS NEEDS TO BE FIXED
    def ComputeSimilarity(self,SEQUENCEONE,SEQUENCETWO):
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
    
    # FIX HOW THE DISTANCE MATRIX IS CALCULATED
    def GenerateTreeForSequence(self):
        ALN=AlignIO.read(StringIO(self.SEQUENCE),'phylip') # INPUTS THE SEQUENCES AS AN ALIGNMENT
        DISTANCEMATRIX=np.zeros([len(ALN), len(ALN)]) # INTIATE DISTANCE MATRIX
        for I,SPECIESONE in enumerate(ALN): # ITERATE THROUGH ALN
            SEQUENCEONE=SPECIESONE.id.split(" ", 1)[1][1:] # GRABBING SEQUENCEONE
            for J,SPECIESTWO in enumerate(ALN): # ITERATE THROUGH ALN
                SEQUENCETWO=SPECIESTWO.id.split(" ", 1)[1][1:] # GRABBING SEQUENCETWO
                if SEQUENCEONE == SEQUENCETWO:
                    DISTANCEMATRIX[I,J] = 0
                else:
                    #DISTANCE=0
                    #for NUCLEOTIDEONE, NUCLEOTIDETWO in zip(SEQUENCEONE, SEQUENCETWO):
                    #    if NUCLEOTIDEONE != NUCLEOTIDETWO:
                    #        DISTANCE += 1
                    #DISTANCEMATRIX[I,J] = DISTANCE
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
    This function generates the maximum time of the tree based on times
    from the taken from the BRANCHLENGTH dictionary and that the first time is t2 or 2.
    '''

    def ReturnTimes(self):
        TIMES=list(self.BRANCHLENGTH.keys()) # CREATES LIST OF CREATES
        COPY=[] # CREATES COPY ARRAY
        for TIME in TIMES: # ITERATE THROUGH BRANCHES
            if "Clade" in str(TIME): # IF CLADE IN THE TIME NAME
                COPY.append(TIME) # APPENDS BRANCHES THAT ARE NOT CLADE TO THE COPY
        LENGTH=len(COPY)+2 # SETS LENGTH TO BE THE MAXIMUM TIME
        return LENGTH # RETURNS MAXIMUM TIME

    '''
    This function generates a mutation dictionary by calculating the 
    number of mutations on a species branch using sequence length being 1000
    and the mutation function.
    '''

    def FindMutations(self):
        MUTATIONDICTIONARY=dict() # CREATES MUTATION DICTIONARY
        SEQUENCELENGTH=self.SEQUENCELENGTH # SEQUENCE LENGTH
        for TAXON in self.BRANCHLENGTH: # ITERATES THROUGH BRANCHES
            if "Clade" not in TAXON: # IF CLADE IS NOT IN THE TAXON NAME
                MUTATION=int(round((SEQUENCELENGTH*self.BRANCHLENGTH[TAXON])/100)) # FINDS AMOUNT OF MUTATIONS ON EACH BRANCH
                MUTATIONDICTIONARY[TAXON]=MUTATION # SETS THE BRANCH VALUE IN DICTIONARY TO THE MUTATION VALUE
        return MUTATIONDICTIONARY # RETURNS THE MUTATION DICTIONARY

    '''
    This function is finding the number of mutations on the branches for all the species and is making sure the species
    are associated with their correct times.
    '''

    # THE FUNCTION SHOULD BE REDUCED TO A SIMPLIFIED FORM
    def FindSpeciesTimes(self,BRANCHLENGTH):
         MUTATIONS=dict() # CREATES A MUTATION DICTIONARY
         TIMES=2 # SETS THE CURRENT TIME TO BE 2
         BRANCHLIST=list(BRANCHLENGTH.keys()) # SETS THE BRANCHLIST TO BE THE LIST OF BRANCHLENGTH KEYS
         BRANCHLIST=list(reversed(BRANCHLIST)) # REVERSED BRANCHLIST
         [BRANCHLIST.remove(CLADE) for CLADE in BRANCHLIST[:] if "Clade" not in CLADE] # ALL BRANCHLIST VALUES THAT ARE NOT THE CLADE
         for CLADE in BRANCHLIST: # ITERATES THROUGH BRANCHLIST
            ONLYTWO=0 # SETS ONLYTWO TO BE 0
            NEXTTWO=False # SETS NEXTTWO TO BE FALSE
            for SPECIES in BRANCHLENGTH: # ITERATES THROUGH BRANCHLENGTH
                if CLADE == SPECIES: # IF THE CLADE == SPECIES
                    NEXTTWO=True # SETS THE NEXT TWO TO BE TRUE
                elif NEXTTWO==True and ONLYTWO<2: # CHECKS IF NEXTTWO = TRUE AND ONLYTWO IS LESS THAN 2
                    if "Clade" not in SPECIES: # IF CLADE IS NOT IN SPECIES
                        MUTATIONS[SPECIES]=TIMES # IT SETS THE SPECIES IN MUTATIONS TO BE THE TIMES VALUE
                    ONLYTWO=ONLYTWO+1 # INCREASES ONLYTWO BY ONE
            TIMES=TIMES+1 # INCREASES TIME BY ONE
         [MUTATIONS.update({CLADE:TIMES}) for CLADE in BRANCHLENGTH if "Clade" not in CLADE and CLADE not in list(MUTATIONS.keys())] # ADDS THE CLADE THAT IS NOT INCLUDED WITH THE MAXIMUM TIME
         DUPLICATEARRAY=[] # DUPLICATE ARRAY
         for MUTATION in list(MUTATIONS.keys()): # ITERATES THROUGH THE MUTATIONS DICTIONARY
            K = [k for k,v in MUTATIONS.items() if v == MUTATIONS[MUTATION]] # FINDING ALL DUPLICATE TIMES
            if K not in DUPLICATEARRAY and len(K) >=2: # SEES IF K IS ALREADY IN THE DUPLICATE ARRAY AND THAT THE LENGTH OF K IS GREATER THAN OR EQUAL TO 2
                DUPLICATEARRAY.append(K) # APPENDS K TO THE DUPLICATE ARRAY
         for DUPLICATEINDEX in range(len(DUPLICATEARRAY)): # ITERATE THROUGH DUPLICATE ARRAY
            for DUPLICATEINDEXONE in range(DUPLICATEINDEX,len(DUPLICATEARRAY)): # ITERATE THROUGH THE VALUES OF THE DUPLICATE ARRAY OF INTEREST
                if(MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]]>MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][0]] and 
                BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEX][0]]<BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEXONE][0]] or
                MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]]<MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][0]] and 
                BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEX][0]]>BRANCHLENGTH[DUPLICATEARRAY[DUPLICATEINDEXONE][0]]):
                   # IF STATEMENT IS CHECKING WHICH OF TWO DUPLICATE CLADE TIMES OCCURS FIRST # EXAMPLE BELOW
                   #print(MUTATIONS)
                   #    -----
                   #  / 
                   #  \
                   #    ------
                   # -|
                   #    ------
                   #  /
                   #  \
                   #    ------
                   EXCHANGEONE=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]] # SETS A VARIABLE EQUAL TO A VALUE IN THE FIRST ARRAY
                   EXCHANGETWO=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][1]] # SETS A VARIABLE EQUAL TO A VALUE IN THE FIRST ARRAY
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][0]]=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][1]] # SWITCHES THE VALUE IN THE FIRST ARRAY WITH THE VALUE IN THE SECOND ARRAY
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEX][1]]=MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][1]] # SWITCHES THE VALUE IN THE FIRST ARRAY WITH THE VALUE IN THE SECOND ARRAY
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][0]]=EXCHANGEONE # GIVES THE VALUE IN THE SECOND ARRAY THE EXCHANGE VALUE
                   MUTATIONS[DUPLICATEARRAY[DUPLICATEINDEXONE][1]]=EXCHANGETWO # GIVES THE VALUE IN THE SECOND ARRAY THE EXCHANGE VALUE
                   #print(MUTATIONS)
         return MUTATIONS # RETURNS THE MUTATIONS DICTIONARY

    '''
    This prior function is used in the non joint and joint likelihood functions.
    '''
    
    def Prior(self,THETA):
        if THETA < 1 and THETA > 0: # PRIOR FUNCTION
            return 1 # RETURNS .5 IF THETA IS BETWEEN THE BOUNDS
        return 0 # RETURNS 0 IF NOT

    '''
    This function returns a likelihood value. It calls the prior function based on theta. It then applied the non joint likelihood function using provided times, theta
    and mutation dictionary. 
    '''

    def JOINTTHETA(self,THETA):
        LP = self.Prior(THETA[0]) # RETURNS A PRIOR BASED ON THETA
        FINAL_VALUE=1 # INITIALIZE FINAL VALUE TO BE 1
        for TIME in range(2,self.TIMES+1): # ITERATE THROUGH TIMES + 1
            BETAVALUE=self.BETA(THETA[0],TIME) # CALCULATE THE BETA VALUE FOR THETA AND TIME
            FINAL_VALUE=FINAL_VALUE*(1/(BETAVALUE+1))*((BETAVALUE/(BETAVALUE+1))**self.MUTATIONDICTIONARY[TIME]) # SETS THE FINAL VALUE TO BE FINAL VALUE * THE JOINT LIKELIHOOD FUNCTION
        return (LP*FINAL_VALUE) # RETURN FINAL VALUE + PRIOR


    '''
    This function return the result of the coalescent time function for
    a provided theta and time value.
    '''
    def TOTALCOALESCENTRATE(self,THETA,CURRENTTIME,TOTALTIPS):
        SUM=0
        LINEAGECOUNT=TOTALTIPS
        for TIME in reversed(range(2,CURRENTTIME+1)):
            SUM=SUM+(-LINEAGECOUNT*(LINEAGECOUNT-1)*mp.exp(THETA/(LINEAGECOUNT*(LINEAGECOUNT-1))))
            LINEAGECOUNT=LINEAGECOUNT-1
            #SUM=SUM+(-NUMBER OF LINEAGES*(NUMBER OF LINEAGES-1)*LENGTH T)
            #SUM=SUM+(TIME*TIME*(TIME-1))
        return SUM
   
    def COALESCENTTIME(self,THETA,TIME,TOTALTIPS):
        return (((TOTALTIPS)*(TOTALTIPS-1))/THETA) * mp.exp((self.TOTALCOALESCENTRATE(THETA,TIME,TOTALTIPS))/THETA)
        #return ((2/THETA)**(TIME-1)) * mp.exp((-2*self.TOTALCOALESCENTRATE(TIME,ANCESTORS))/THETA)
        #return ((2/THETA)**(TIME-1)) * mp.exp((-2*self.TOTALCOALESCENTRATE(TIME))/THETA)
        #return ((THETA)/(TIME*(TIME-1))) # COALESCENT TIME FUNCTION

    def MUTUTATIONTIMES(self,THETA,MUTATION,LINEAGES):
        #if MUTATION == 0:
        #    return 0
        #print(MUTATION)
        # 4*1000 MAY NEED TO BE FIXED
        return (mp.exp(((THETA/(4*self.SEQUENCELENGTH))* LINEAGES * mp.exp(THETA/(LINEAGES*(LINEAGES-1))))*((THETA/(4*self.SEQUENCELENGTH)) * LINEAGES * mp.exp(THETA/(LINEAGES*(LINEAGES-1)))))/math.factorial(MUTATION))
    '''
    This function returns a likelihood value. It calls the prior function based on theta. It then applied the joint likelihood function using provided times, theta
    and mutation dictionary. 
    '''
    
    def JOINTTHETATIMES(self,THETA):
        LP = self.Prior(THETA[0]) # RETURNS A PRIOR BASED ON THETA
        FINALVALUE=1 # INITIALIZE FINAL VALUE TO BE 1
        NUMBEROFLINEAGES=self.TIMES # NUMBER OF LINEAGES
        for TIME in range(2,self.TIMES+1): # ITERATE THROUGH TIMES + 1
            #POWER=(1*TIME*self.COALESCENTTIME(THETA[0],TIME))
            #print(self.MUTUTATIONTIMES(THETA[0],self.MUTATIONDICTIONARY[TIME][0],TIME,NUMBEROFLINEAGES))
            #print((self.COALESCENTTIME(THETA[0],TIME,NUMBEROFLINEAGES,self.TIMES)))
            FINALVALUE=FINALVALUE*((self.MUTUTATIONTIMES(THETA[0],self.MUTATIONDICTIONARY[TIME][0],NUMBEROFLINEAGES))*(self.COALESCENTTIME(THETA[0],TIME,self.TIMES))) # SETS THE FINAL VALUE TO BE FINAL VALUE * THE NON JOINT LIKELIHOOD FUNCTION
            NUMBEROFLINEAGES=NUMBEROFLINEAGES-1
        return (LP * FINALVALUE) # RETURN FINAL VALUE + PRIOR
        #return THETA * 2
    
    '''
    This function generates a number of values from a log norm distribution based on the number of walkers. It then runs the 
    emcee sampler for both the joint likelihood and non joint likelihood functions. It then grabs the chain where which is then used as
    a return value and shown in a plot.
    '''

    def MHastings(self,DICTIONARY):
            print("New Topology")
            print(DICTIONARY) # PRINTS THE NEW DICTIONARY TOPOLOGY
            self.MUTATIONDICTIONARY=DICTIONARY # SETS THE MUTATION DICTIONARY TO BE THIS NEW DICTIONARY TOPOLOGY
            NWALKER=self.WALKERS # SETS THE NUMBER OF WALKERS
            NDIM=1 # SETS THE NUMBER OF DIMENSIONS
            POSITIONJOINT = [lognorm.rvs(self.SHAPE, size=1) for i in range(NWALKER)] # GENERATES A BUNCH OF VALUES FROM A LOG NORM DISTRIBUTION BASED ON THE NUMBER OF WALKERS
            POSITIONNONJOINT = POSITIONJOINT.copy() # SETS POSITIONNONJOIN TO BE EQUAL TO A COPY  OF POSITIONJOINT
            JOINTSAMPLER = emcee.EnsembleSampler(NWALKER,NDIM,self.JOINTTHETA,args=()) # CREATES A JOINT SAMPLER OBJECT
            JOINTSAMPLER.run_mcmc(POSITIONJOINT,1) # RUNS THE JOINT SAMPLER OBJECT
            NONJOINTSAMPLER = emcee.EnsembleSampler(NWALKER,NDIM,self.JOINTTHETATIMES,args=()) # CREATES A NON JOINT SAMPLER OBJECT
            NONJOINTSAMPLER.run_mcmc(POSITIONNONJOINT,1) # RUNS THE NON JOINT SAMPLER OBJECT
            samplesone = JOINTSAMPLER.get_chain(flat=True) # SETS A VARIABLE EQUAL TO THE MATRIX CHAIN FOR THE JOINT SAMPLER
            samplestwo = NONJOINTSAMPLER.get_chain(flat=True) # SETS A VARIABLE EQUAL TO THE MATRIX CHAIN FOR THE NON JOINT SAMPLER
            print("Joint Sampler")
            #print(samplesone[:, 0])
            print("Non Joint Sampler")
            #print(samplestwo[:, 0])
            #plt.hist(samplesone[:, 0], 100, color="k", histtype="step")
            #plt.hist(samplestwo[:, 0], 100, color="m", histtype="step")
            #plt.xlabel(r"$\theta_1$")
            #plt.ylabel(r"$p(\theta_1)$")
            #plt.gca().set_yticks([])
            #plt.show()
            return samplesone[:, 0],samplestwo[:, 0] # RETURNS THE CHAINS FOR THE JOINT AND NON JOINT SAMPLER
   
    '''
    This function is a recursive function that is iterating through the carry mutations and placing them into the passdictionary at
    a time and determining the number of mutations that should be passed on to the next time. There is not a result but it is appending each different
    pass dictionary into the species dictionary
    '''
    
    def callBackMutation(self,TIME,CURRENTTIME,SPECIE,PASSDICTIONARY,CARRY,MUTATION):
        if CURRENTTIME-1<TIME: # CURRENT TIME MUST BE LESS THAN THE MAXIMUM TIME
            for TIMEPLACEMENT in range(1, CARRY+2): # ITERATE THROUGH THROUGH CARRY MUTATIONS
                PASSDICTIONARY[CURRENTTIME] = TIMEPLACEMENT-1 # SETS THE MUTATION NUMBER AT CURRENT TIME
                CARRYOVER=CARRY-PASSDICTIONARY[CURRENTTIME] # CREATES A CARRYOVER VARIABLE BASED ON THE OVERALL MUTATIONS SUBTRACTED BY THE MUTATIONS FOR THE FIRST TIME
                self.callBackMutation(TIME,CURRENTTIME+1,SPECIE,PASSDICTIONARY,CARRYOVER,MUTATION) # CALLS THE MUTATION FUNCTION FOR THE NEXT TIME
        elif CURRENTTIME-1==TIME: # IF CURRENT TIME - 1 IS EQUALED TO TIME
            PASSDICTIONARY[CURRENTTIME] = CARRY # SETS THE TIME IN THE DICTIONARY TO BE THE MUTATION PLACES AT THIS TIME
            self.SPECIESDICTIONARY[SPECIE].append(PASSDICTIONARY.copy()) # APPENDS DICTIONARY INTO SPECIES DICTIONARY
    
    '''
    This is a recursive function that is determining the mutation placements for all the species. It calls the CallBackMutations function.
    '''

    def callBacks(self,CALLBACKS,MUTATIONS,SPECIETIMES):
        for SPECIE in CALLBACKS: # ITERATING THROUGH SPECIES
            MUTATION = MUTATIONS[SPECIE] # CREATEING MUTATION VARIABLE FOR SPECIE
            TIME = SPECIETIMES[SPECIE] # CREATING TIME VARIABLE FOR SPECIE
            CURRENTTIME = 2 # SET CURRENT TIME TO 2
            self.SPECIESDICTIONARY[SPECIE] = [] # SET SPECIES KEY IN DICTIONARY TO ARRAY
            PASSDICTIONARY = {} # CREATE DICTIONARY
            for FIRSTTIME in range(1, int(MUTATION)+2): # ITERATE THROUGH MUTATIONS
                PASSDICTIONARY[CURRENTTIME] = FIRSTTIME-1 # SET TIME IN PASSDICTIONARY TO MUTATION VALUE
                CARRYOVER = MUTATION-(PASSDICTIONARY[CURRENTTIME]) # DETERMINE THE MUTATIONS TO CARRY TO THE NEXT SPECIES
                self.callBackMutation(TIME, CURRENTTIME+1, SPECIE, PASSDICTIONARY, CARRYOVER, MUTATION) # CALLS CALLBACKMUTATIONS FUNCTION

    '''
    This function is recursive function that is creating an array based on the combinations of the mutation placements for all the species. M=
    '''

    def IterateThroughIntervalDictionary(self,INTERVALDICTIONARY,KEYS,INDEX,TIMEDICTIONARY):
        if INDEX < len(KEYS): # ITERATES THROUGH THE KEYS 
            ARRAYS=INTERVALDICTIONARY[KEYS[INDEX]] # SETS THE ARRAYS VARIABLE TO THE DICTIONARY FOR ONE OF THE SPECIES
            for ARRAY in ARRAYS: # ITERATE THROUGH ARRAYS
                COPY=TIMEDICTIONARY.copy() # CREATES A TIMEDICTIONARY COPY
                for MINIARRAY in list(reversed(ARRAY)): # ITERATES THROUGH THE DICTIONARY
                    #if INDEX + 1 < len(KEYS)-1:
                        VALUE=sum(COPY[MINIARRAY]) # TAKES THE SUM FOR THE COPY DICTIONARY AT A KEY -- TIME
                        COPY[MINIARRAY] = [VALUE + ARRAY[MINIARRAY]] #SETS THE COPY AT A GIVEN TIME TO BE EQUAL TO THE SUM VALUE AT TIME + CURRENT VALUE
                self.IterateThroughIntervalDictionary(INTERVALDICTIONARY,KEYS,INDEX+1,COPY)  # RECALLS ITSELF USING A INDEX+1 AND COPY
        else:
            #print(sum(list(self.MUTATIONS.values())))
            #print(TIMEDICTIONARY)
            #print(self.MUTATIONS)
            SUMTIMEMUTATIONS=sum(TIMEDICTIONARY[MUTATION][0] for MUTATION in TIMEDICTIONARY) # ITERATES THROUGH THE TIMEDICTIONARY AND ATTAINS THE SUM MUTATION VALUE
            #print(SUMTIMEMUTATIONS)
            #print(sum(self.MUTATIONS.values()))
            if TIMEDICTIONARY not in self.FINALTIMEARRAY and SUMTIMEMUTATIONS==sum(self.MUTATIONS.values()): # IF STATEMENT CHECKING IF TIMEDICTIONARY IS NOT IN THE FINAL ARRAY AND SUMTIMEMUTATIONS IS EQUAL TO SUM OF THE MUTATIONS
                self.FINALTIMEARRAY.append(TIMEDICTIONARY.copy()) # APPENDS INTO FINAL ARRAY
  
    '''
    This function runs a good majority of the functions written earlier. It then calls Metropolis Hastings
    on all the mutation topologies are different topologies  -- we consider the swapping of inner and outer clades
    '''  

    def MHastingsLoop(self):
        self.TIMES=self.ReturnTimes() # RETURNS TIME
        self.MUTATIONS=self.FindMutations() # RETURNS MUTATIONS
        self.SPECIETIMES=self.FindSpeciesTimes(self.BRANCHLENGTH) # RETURNS SPECIES TIMES
        print(self.MUTATIONS)
        CALLBACKS=[[]] # CREATES A CALLBACK ARRAY
        for TAXON in self.BRANCHLENGTH:
            if "Clade" not in TAXON: # WORK IN PROGRESS
                CALLBACKS[0].append(TAXON) # WORK IN PROGRESS
        CALLBACKS.append(CALLBACKS[0][2:4]+CALLBACKS[0][0:2]) # THE IDEA OF THIS LINE AND THE LINE ABOVE IS THAT IT SHOULD BE APPENDING ALL 
        # POSSIBLE TOPOLOGIES? SWITCHING THE INNER AND OUTER CLADES? STILL A WORK IN PROGRESS BUT RIGHT NOW THIS FUNCTION IS ONLY TESTING ON
        # TWO POSSIBLE TOPOLOGIES INNER AND OUT CLADES BEING SWAPPED
        for CALLBACKINDEX in range(len(CALLBACKS)): # ITERATES THROUGH CALLBACK
            self.SPECIESDICTIONARY=dict() # CREATES SPECIES DICTIONARY
            self.callBacks(CALLBACKS[CALLBACKINDEX],self.MUTATIONS,self.SPECIETIMES) # CALLS CALL BACK FUNCTION
            KEYS=list(self.SPECIESDICTIONARY.keys()) # CREATES A LIST OF THE SPECIES DICTIONARY KEYS
            TIMEDICTIONARY=dict.fromkeys(range(2,self.TIMES+2), [0]) # CREATES A DICTIONARY USING TIMES
            self.IterateThroughIntervalDictionary(self.SPECIESDICTIONARY,KEYS,0,TIMEDICTIONARY) # CALLS ITERATE THROUGH INTERVAL DICTIONARY FUNCTION
            for DICTIONARY in self.FINALTIMEARRAY: # ITERATES THROUGH FINALTIMEARRAY
                DICTIONARY[self.TIMES] = [DICTIONARY[self.TIMES][0] + DICTIONARY[self.TIMES+1][0]] # SETS DICTIONARY AT TIME TO BE EQUAL AT CURRENT TIME + 1
                DICTIONARY[self.TIMES+1] = [0] # SETS THE LAST TIME TO BE 0
                # ABOVE IS DONE BECAUSE OF SOME ISSUE WHERE YOU END UP WITH ONE MORE TIME THEN NECESSARY
                # EXAMPLE IF TIMES ARE 4
                # 2 3 4 ARE THE TIME INTERVALS YOU SHOULD END UP WITH 2 3 4 5 THUS WE ARE GETTING RID OF THE 5 
            self.FINALTIMEARRAY = [i for n, i in enumerate(self.FINALTIMEARRAY) if i not in self.FINALTIMEARRAY[n + 1:]] # CREATES FINAL TIME ARRAY # GETTING RID OF DUPLICATES
            # ITERATE THROUGH FINAL TIME ARRAY
            for DICTIONARYINDEX in range(len(self.FINALTIMEARRAY)):
               self.MHASTINGPROBS[DICTIONARYINDEX] = self.MHastings(self.FINALTIMEARRAY[DICTIONARYINDEX]) # CALLS METROPOLIS HASTINGS AND APPENDS RESULTS TO A DICTIONARY
            self.TREEPROBS[CALLBACKINDEX]=dict(self.MHASTINGPROBS.copy()) # APPENDS THE MHASTINGS DICTIONARY TO A NEW DICTIONARY AT CALLBACKINDEX
            self.MHASTINGPROBS=dict() # RESETS MHASTINGPROB DICTIONARY

    '''
    This function returns the distribution of the theta values produced by metropolis hastings.
    '''

    def CombineSamples(self):
        JOINTTOTAL=np.zeros(shape=(0, 0)) # CREATES AN EMPTY NP ARRAY
        NONJOINTTOTAL=np.zeros(shape=(0,0)) # CREATES AN EMPTY NP ARRAY
        for POSSIBLETOPOLOGIES in self.TREEPROBS.keys(): # ITERATE THROUGH THE TREE TOPOLOGIES 
            for POSSIBLEMUTATIONTOPOLOGIES in self.TREEPROBS[POSSIBLETOPOLOGIES].keys(): # ITERATES THROUGH THE MUTATION TOPOLOGIES
                JOINTTOTAL=np.concatenate((JOINTTOTAL,self.TREEPROBS[POSSIBLETOPOLOGIES][POSSIBLEMUTATIONTOPOLOGIES][0].reshape(-1,self.WALKERS)),axis=None) # CONCATS THE JOINT LIKELIHOOD THETAS TO AN A JOINT LIKELIHOOD THETA ARRAY
                NONJOINTTOTAL=np.concatenate((NONJOINTTOTAL,self.TREEPROBS[POSSIBLETOPOLOGIES][POSSIBLEMUTATIONTOPOLOGIES][1].reshape(-1,self.WALKERS)),axis=None) # CONCATS THE NON JOINT LIKELIHOOD THETAS TO AN A JOINT LIKELIHOOD THETA ARRAY
        NJSET=np.unique(NONJOINTTOTAL) # TAKES ALL UNIQUE THETA VALUES FOR NON JOINT LIKELIHOOD
        JSET=np.unique(JOINTTOTAL) # TAKES ALL UNIQUE THETA VALUES FOR JOINT LIKELIHOOD
        #print("Non Joint Sampler Unique Sum") # PRINT STATEMENT
        #print(NJSET) # PRINT STATEMENT
        #print("Joint Sampler Unique Sum") # PRINT STATEMENT
        #print(JSET) # PRINT STATEMENT
        #plt.hist(NJSET,bins=15, alpha=0.4, label='NON JOINT LIKELIHOOD THETA DISTRIBUTION',density=True) # PLOT
        #plt.hist(JSET,bins=15, alpha=0.4, label='JOINT LIKELIHOOD THETA DISTRIBUTION',density=True) # PLOT
        #plt.legend(loc='upper right') # PLOT
        #plt.xlim([-25, 25]) # PLOT
        #plt.show() # PLOT
        #KDENJ = KernelDensity(bandwidth=1, kernel='gaussian')
        #KDENJ.fit(NJSET.reshape(-1, 1))
        #KDEJ = KernelDensity(bandwidth=1, kernel='gaussian')
        #KDEJ.fit(JSET.reshape(-1, 1))
        #XNJ = np.linspace(np.min(NJSET.reshape(-1, 1)), np.max(NJSET.reshape(-1, 1)))
        #XJ = np.linspace(np.min(JSET.reshape(-1, 1)), np.max(JSET.reshape(-1, 1)))
        #LOGPROBNJ = np.exp(KDENJ.score_samples(XNJ.reshape(-1,1)))
        #LOGPROBJ = np.exp(KDEJ.score_samples(XJ.reshape(-1,1)))
        sns.kdeplot(JSET.reshape(-1,1),label='JOINT LIKELIHOOD THETA')
        #plt.plot(XJ.reshape(-1,1), LOGPROBJ.reshape(-1,1))
        #plt.legend()
        #plt.xlim([-25, 25])
        #plt.show()
        sns.kdeplot(NJSET.reshape(-1,1),label='NON JOINT LIKELIHOOD THETA',color=(""))
        plt.gca().get_lines()[0].set_color("red")
        plt.gca().get_lines()[1].set_color("blue")
        #plt.plot(XNJ.reshape(-1,1), LOGPROBNJ.reshape(-1,1))
        plt.legend(loc='upper right')
        plt.xlim([-25, 25])
        plt.show()
        

'''
Calls the TreeSequenceGeneration class and runs 
the GenerateTheta, SimulateTree, SimulateSequence,
GenerateTreeForSequence, ReturnBranchLengths, and
MHastingsLoop functions.
'''

if __name__ == "__main__":
    TSG=TreeSequenceGeneration(1,100)
    TSG.GenerateTheta()
    TSG.SimulateTree()
    TSG.SimulateSequence()
    TSG.GenerateTreeForSequence()
    TSG.ReturnBranchLengths()
    TSG.MHastingsLoop()
    TSG.CombineSamples()






