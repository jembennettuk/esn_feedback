###
### Specify input/model/training/testing parameters
###
import numpy as np

###
### Input data
###
inputData = 'seqMNIST' # Keyword for input/label data
inputDir = './data/'    # Storage directory for input/label data

###
### Training properties
###
nEpochs=5000#5000 # No optimiser iterations
batch_size=50 # No. input samples per optimser iteration
etaInitial = 0.0001 # Optimser learning rate for initial (e.g. metric) learning phase
etaTransfer = 0.001 # Optimser learning rate for transfer learning phase
eta_tau = np.Inf#5000.0 # Time constant for learning rate decay
nClass=10 # No. classes in dataset
tMax = 28 # No. time steps per input sample
tauPerf = tMax / 2 # Time scale for weighting responses for accuracy/loss

###
### ESN properties
###
esnDir = '.' # Directory to store ESN response
alpha = 0.1  # Decay rate of ESN units [0,1]
rho = 0.99   # Scale ESN-ESN weights
gamma = 0.1  # Scale input-ESN weights
N_av = 10    # Fan-out no. from ESN units
N_esn = 500      # No. of ESN units
nInputs = 28      # No. input features
esn_tau = 1.0 / alpha / (1.0 - rho) # Maimum timesclae of ESN dynamics
fbLayer = 2 # Feedback layer

###
### Metric (MET) learning properties
###
METflag = True               # Include MET layer
METclass = 'METlin'          # MET layer model class [METlin - linear, METleaky - leaky integrator]
METsave = True               # Save MET validation responses for analyses
METsaveN = 100               # No. of save-points throughout MET layer training.
# metricLossType = 'tripletLoss' # Training method for metric learning [triplet; HalvagalZenke]
metricLossType = 'prototypicalLoss' # Training method for metric learning [triplet; HalvagalZenke]
nSampProto = 20              # Np. samples to compute prototypes
eta_met = 0.0001             # Learning rate for ESN-MET weights
margin=2                     # For triplet margin loss
tranFromLayer = 3            # Perform transfer learning from this layer

###
### Classification learning properties
###
OUTsave = False         # Save output validation responses for analyses 
nEpisodesOut = 2000     # No. training batches
nTransferRuns = 1       # No. instances of transfer learning per learned hidden network
N_check = 50            # No. validation checks throughout training
eta_out = 0.01             # Learning rate for ESN-OUT/MET-OUT weights

###
### Feedforward properties
###
Ns = [N_esn, 200, 200, 50, nClass] # No. neurons in each layer

###
### Readout and save properties
###
outsPerTime = True # Specifiy whether use a readout weight per time point per class (True)
                    # or just one weight per class (False)
reportTime = 500 # Real time Accuracy/Loss report every reportTime time steps
saveLayers = [1, 2] # Save responses from these layers
saveRespAtN = [] # Save response at these time points
nSaveMaxT = nEpochs #5000 # Save responses up to this iteration
nSaveSamples = 20 # No. samples to save per class
saveFlag_RESP = True # Save responses
saveFlag_FBWeights = False if not fbLayer else True # Save feedback weights
saveFlag_DW = True # Save histograms of weight changes
nWeightSave = 25 # No. time points to save feedback weights

###
### Validation
###
N_val = 10000 # No. data samples for validation (taken from training set)
