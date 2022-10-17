#IMPORTS
from re import S
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
np.random.seed(123)

class TreeSequenceGeneration():
    
    #VARIABLES THAT ARE USED IN DIFFERENT FUNCTIONS
    SHAPE=""
    THETA=""
    TREE=""
    SEQUENCE=""
    SEQSEQ=""
    TAXA=""
    POPULATIONSIZE=""
    COALESCENTTTIME=""
    GENERATED_TREE=""
    TIMES=""
    BRANCHLENGTH=dict()
    TREESTRING=""
    STRINGUSEDDICTIONARY=dict()
    JFINAL=[]
    NJFINAL=[]
    ALLNEWICKSTRING=[]
    SPECIESDICTIONARY=dict()
    FINALTIMEARRAY=[]

    def __init__(self,SHAPE)->None:
        pass 
        self.SHAPE=SHAPE # PASS A SHAPE WHEN CREATING OBJECT
    
    def GenerateTheta(self):
        self.THETA=lognorm.rvs(self.SHAPE, size=1) # GENERATE THETA FROM LOG NORM DISTRIBUTION SHAPE REPRESENTS THE MAX VALUE THAT CAN BE DRAWN
        return self.THETA
    
    def SimulateTree(self):
        self.TAXA=dendropy.TaxonNamespace(["z1", "z2", "z3"]) # AMOUNT OF TAXAS/SPECIES
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
    
    def SimulateSequence(self):
        self.SEQUENCE=dendropy.model.discrete.hky85_chars(1000,tree_model=self.TREE) # SIMULATE SEQEUNCES OF THE TREE UP TO 1000 CHARACTERS
        self.SEQUENCE=self.SEQUENCE.as_string(schema="phylip") # SHOW SEQUENCE USING THE PHYLIP SCHEME
        print("Sequence Derived from Tree Simulation")
        print()
        print(self.SEQUENCE)
        print("--------------------------------------------------------------------------------------------------")
        print()
    
    def ComputeSimilarity(self,SEQUENCEONE,SEQUENCETWO):
        # HAVE TO ACCOMODATE FOR GAPS INSERTIONS AND DELETIONS LATER ON SO WE HAVE TO DO SOME FORM OF ALIGNMENT TO COMPUTE SIMILARLY MAYBE?
        # MAYBE NOT AS THIS IS BASED MORE ON THE TERMS SIMILARLY
        TERMS=set(SEQUENCEONE).union(SEQUENCETWO) # TAKES THE UNION OF THE TWO SETS
        DOTPRODUCT=sum(SEQUENCEONE.get(k,0)*SEQUENCETWO.get(k,0)for k in TERMS) # MULTIPLY OCCURENCES OF EACH NUCLEOTIDE IN SEQUENCEONE BY SEQUENCETWO THEN TAKE SUM
        MAGA=math.sqrt(sum(SEQUENCEONE.get(k,0)**2 for k in TERMS)) # TAKE THE OCCURENCE OF EACH NUCLEOTIDE SQUARED THEN TAKE SUM SEQUENCEONE
        MAGB=math.sqrt(sum(SEQUENCETWO.get(k,0)**2 for k in TERMS)) # TAKE THE OCCURENCE OF EACH NUCLEOTIDE SQUARED THEN TAKE SUM SEQUENCETWO
        return DOTPRODUCT/(MAGA*MAGB) 
    
    #def PARTITIONSET(self,SEQUENCE):
    #    if len(SEQUENCE)==1:
    #        yield frozenset(SEQUENCE)
    #        return
    #    ELEM,*_=SEQUENCE
    #    REST=frozenset(SEQUENCE-{ELEM})
    #    for PARTITION in self.PARTITIONSET(REST):
    #        for SUBSET in PARTITION:
    #            try:
    #                ASUBSET=frozenset(SUBSET|frozenset({ELEM}))
    #            except TypeError:
    #                ASUBSET=frozenset({SUBSET}|frozenset({ELEM}))
    #            yield frozenset({ASUBSET})|(PARTITION-{SUBSET})
    #        yield frozenset({ELEM})|PARTITION
    #def PRINTSET(self,f):
    #    if type(f) not in (set, frozenset):
    #        return str(f)
    #    return "(" + ",".join(sorted(map(self.PRINTSET, f))) + "):0.0"
    
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

        #FINDCLADE=re.findall("[),]:[0-9].[0-9]+",UPDATEDDISTANCEMATRIX)
        #for NEWICK in self.PARTITIONSET(set(re.findall("[a-zA-Z].:[0-9].[0-9]+",UPDATEDDISTANCEMATRIX))):
        #    STRING=self.PRINTSET(NEWICK) + ";"
        #    for CLADE in FINDCLADE:
        #        STRING=STRING.replace("0.0",CLADE[2:],1)
        #        if CLADE[2:]+";" in STRING:
        #            STRING=STRING.replace(CLADE[2:],"0.0",1)
        #    self.ALLNEWICKSTRING.append(STRING)
        #self.ALLNEWICKSTRING=self.ALLNEWICKSTRING[1:]
        #self.TREESTRING=str(upgma(DISTANCEMATRIX))
        #print(UPDATEDDISTANCEMATRIX)

        self.GENERATED_TREE=Phylo.read(HANDLE, "newick") # READS THE NEWICK STRING AS A TREE
        self.GENERATED_TREE.rooted=True # SET THE TREE TO BE ROOTED
        self.GENERATED_TREE.ladderize() # REORGANIZES TREE
        print("Generated Tree from Sequences")
        print()   
        Phylo.draw_ascii(self.GENERATED_TREE) # DRAWS TREE
        print("--------------------------------------------------------------------------------------------------")
        print()

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

    def BETA(self,THETA,TIME):
        return THETA/(TIME-1) # BETA FUNCTION

    def COALESCENTTIME(self,THETA,TIME):
        return ((THETA)/(TIME*(TIME-1))) # COALESCENT TIME FUNCTION

    def ReturnTimes(self):
        TIMES=list(self.BRANCHLENGTH.keys()) # CREATES LIST OF CREATES
        COPY=[] # CREATES COPY ARRAY
        for TIME in TIMES: # ITERATE THROUGH BRANCHES
            if "Clade" in str(TIME):
                COPY.append(TIME) # APPENDS BRANCHES THAT ARE NOT CLADE TO THE COPY
        LENGTH=len(COPY)+2 # RETURNS MAXIMUM TIME
        return LENGTH

    def FindMutations(self):
        MUTATIONDICTIONARY=dict() # CREATES MUTATION DICTIONARY
        SEQUENCELENGTH=1000 # SEQUENCE LENGTH
        for TAXON in self.BRANCHLENGTH: # ITERATES THROUGH BRANCHES
            if "Clade" not in TAXON: 
                MUTATION=int(round((SEQUENCELENGTH*self.BRANCHLENGTH[TAXON])/100)) # FINDS AMOUNT OF MUTATIONS ON EACH BRANCH
                MUTATIONDICTIONARY[TAXON]=MUTATION # SETS THE BRANCH VALUE IN DICTIONARY TO THE MUTATION VALUE
        return MUTATIONDICTIONARY

    # MAYBE SIMPLIFY CODE OR REWRITE SO IT DOES NOT USE AS MANY FOR LOOPS?
    def FindSpeciesTimes(self,BRANCHLENGTH):
         MUTATIONS=dict()
         TIMES=2
         BRANCHLIST=list(BRANCHLENGTH.keys())
         BRANCHLIST=list(reversed(BRANCHLIST))
         for CLADE in BRANCHLIST[:]:
            if "Clade" not in CLADE:
                BRANCHLIST.remove(CLADE)
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
         for CLADE in BRANCHLENGTH:
            if "Clade" not in CLADE and CLADE not in list(MUTATIONS.keys()):
                MUTATIONS[CLADE]=TIMES
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

    def Prior(self,THETA):
        return 1 # PRIOR FUNCTION IN PROGRESS

    def LLNJ(self,THETA):
        #I NEED A LOG PRIOR
        FINAL_VALUE=1
        for TIME in range(2,self.TIMES+1):
            BETAVALUE=self.BETA(THETA[0],TIME)
            FINAL_VALUE=FINAL_VALUE*(1/(BETAVALUE+1))*((BETAVALUE/(BETAVALUE+1))**1)
            # WILL LOGO PRIOR BE CALLED ON THETA THEN ADDED WITH THE FINAL VALUE
        return FINAL_VALUE

    def LLJ(self,THETA):
        #I NEED A LOG PRIOR
        FINALVALUE=1
        for TIME in range(2,self.TIMES+1):
            POWER=(1*TIME*self.COALESCENTTIME(THETA[0],TIME))
            FINALVALUE=FINALVALUE*((POWER)*(self.COALESCENTTIME(THETA[0],TIME)))
        return FINALVALUE
    
    def MHastings(self,DICTIONARY):
            # UPDATE LIKELIHOOD FUNCTIONS WITH DICTIONARY VALUE
            NWALKER=2
            NDIM=1
            POSITIONJOINT = [lognorm.rvs(self.SHAPE, size=1) for i in range(NWALKER)]
            POSITIONNONJOINT = POSITIONJOINT
            JOINTSAMPLER = emcee.EnsembleSampler(NWALKER,NDIM,self.LLNJ,args=())
            JOINTSAMPLER.run_mcmc(POSITIONJOINT,1)
            NONJOINTSAMPLER = emcee.EnsembleSampler(NWALKER,NDIM,self.LLJ,args=())
            NONJOINTSAMPLER.run_mcmc(POSITIONNONJOINT,1)   
    
    def callBackMutation(self,ROOTPLACEMENT,TIME,CURRENTTIME,SPECIE,PASSDICTIONARY):
        if CURRENTTIME-1<TIME:
            for TIMEPLACEMENT in range(1,ROOTPLACEMENT+1):
                PASSDICTIONARY[TIME]=TIMEPLACEMENT-1
                self.callBackMutation(TIMEPLACEMENT,TIME-1,CURRENTTIME,SPECIE,PASSDICTIONARY)
        elif CURRENTTIME-1==TIME:
            if PASSDICTIONARY not in self.SPECIESDICTIONARY[SPECIE]:
                    self.SPECIESDICTIONARY[SPECIE].append(PASSDICTIONARY.copy())
    
    def callBacks(self,CALLBACKS,MUTATIONS,SPECIETIMES):
       if CALLBACKS != []:
            LIST=list(reversed(CALLBACKS))
            SPECIE=LIST[0]
            LIST.remove(SPECIE)
            LIST=list(reversed(LIST))
            MUTATION=MUTATIONS[SPECIE]
            TIME=SPECIETIMES[SPECIE]
            CURRENTTIME=2
            self.SPECIESDICTIONARY[SPECIE]=[]
            for FIRSTTIME in range(1,int(MUTATION)+2):
                PASSDICTIONARY=dict()
                PASSDICTIONARY[TIME]=FIRSTTIME-1
                if SPECIETIMES[SPECIE] == 2:
                    self.SPECIESDICTIONARY[SPECIE].append(PASSDICTIONARY.copy())
                self.callBackMutation(FIRSTTIME,TIME-1,CURRENTTIME,SPECIE,PASSDICTIONARY)
            if LIST != []:
                return self.callBacks(LIST,MUTATIONS,SPECIETIMES)

    def IterateThroughIntervalDictionary(self,INTERVALDICTIONARY,KEYS,INDEX,TIMEDICTIONARY):
        if INDEX < len(KEYS):
            ARRAYS=INTERVALDICTIONARY[KEYS[INDEX]]
            for ARRAY in ARRAYS:
                COPY=TIMEDICTIONARY.copy()
                for MINIARRAY in ARRAY:   
                    VALUE=sum(TIMEDICTIONARY[MINIARRAY])
                    COPY[MINIARRAY]=[VALUE+ARRAY[MINIARRAY]]
                if COPY not in self.FINALTIMEARRAY:
                    self.FINALTIMEARRAY.append(COPY.copy())
                self.IterateThroughIntervalDictionary(self.SPECIESDICTIONARY,KEYS,INDEX+1,COPY) 
            
    def MHastingsLoop(self):
        self.TIMES=self.ReturnTimes()
        self.MUTATIONS=self.FindMutations()
        self.SPECIETIMES=self.FindSpeciesTimes(self.BRANCHLENGTH)
        CALLBACKS=[]
        for TAXON in self.BRANCHLENGTH:
            if "Clade" not in TAXON:
                CALLBACKS.append(TAXON)
        self.callBacks(CALLBACKS,self.MUTATIONS,self.SPECIETIMES)
        KEYS=list(self.SPECIESDICTIONARY.keys())
        TIMEDICTIONARY=dict.fromkeys(range(2,self.TIMES+1), [0])
        self.IterateThroughIntervalDictionary(self.SPECIESDICTIONARY,KEYS,0,TIMEDICTIONARY)
        for DICTIONARY in self.FINALTIMEARRAY:
            self.MHastings(DICTIONARY)

# RUNS FUNCTIONS IN THE CLASS
if __name__ == "__main__":
    TSG=TreeSequenceGeneration(1)
    TSG.GenerateTheta()
    TSG.SimulateTree()
    TSG.SimulateSequence()
    TSG.GenerateTreeForSequence()
    TSG.ReturnBranchLengths()
    TSG.MHastingsLoop()






