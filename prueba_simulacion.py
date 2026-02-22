from functions import *

path = '/lustre/LHCb/alejandro.rodriguez/DecFiles/DVntuple.root'
tree = 'MCDecayTreeTuple/MCDecayTree;1'
columns = [
    "KL0_MC_MOTHER_ID",
    "KL0_MC_MOTHER_KEY",
    "KL0_MC_GD_MOTHER_ID",
    "KL0_MC_GD_MOTHER_KEY",
    "KL0_MC_GD_GD_MOTHER_ID",
    "KL0_MC_GD_GD_MOTHER_KEY",
    "KL0_TRUEP_E",
    "KL0_TRUEP_X",
    "KL0_TRUEP_Y",
    "KL0_TRUEP_Z",
    "KL0_TRUEPT",
    "KL0_TRUEORIGINVERTEX_X",
    "KL0_TRUEORIGINVERTEX_Y",
    "KL0_TRUEORIGINVERTEX_Z",
    "KL0_TRUEENDVERTEX_X",
    "KL0_TRUEENDVERTEX_Y",
    "KL0_TRUEENDVERTEX_Z",
    "KL0_TRUEISSTABLE",
    "KL0_TRUETAU",
    "nCandidate",
    "totCandidates",
    "EventInSequence"
]


df = read_root(path, tree, columns)

print(df.head())