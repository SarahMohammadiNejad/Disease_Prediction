import numpy as np

class Simulation:
    def __init__(self,dataset):
        self.saved_models = {}
        self.saved_model_labels = ['lr','svcl0','svcg0','svcl','svcg','dt','rf','nn']
        self.models2train = {}
        self.models2showResult = {}
        self.models2predict = {}

        self.dataset = dataset
        self.num_disease = 1

        self.DoTrain = True #Training is going to be done?
        self.DoTrai = {} # A list that shows which models are going to be trained
    
        self.DoPredInput = True 
        self.DoPredInpu = {}

        self.DoShowRes = {} # A list that shows which models are going to be trained
        self.DoShowRes['lr'] = True

        self.dropout_variables = []
        self.categorical_variables = []
        self.dependent_variables = []
        self.ordinal_variables = []

        self.Create_EDAplot = False
        self.DoPlot_ConfusionM = False#True
        self.DoWrite_metrics = False#True
        self.DoPlot_NNloss = False
        self.DoPlot_FeatureImp = False#True


    def dataset_param(self):
        if self.dataset == 'Clinical':
            self.num_disease = 1
        else:
            self.num_disease = 5

    def task_param(self):
        self.DoTrai['lr'] = False #True#False #True#False#True
        self.DoTrai['svcl'] = False #True#False #True#False#True#True #Calibrated
        self.DoTrai['svcg'] = False #True#False #True#True#False#True #Calibrated
        self.DoTrai['svcl0'] = False #Normal
        self.DoTrai['svcg0'] = False #Normal
        self.DoTrai['dt'] = False #True#False #True#True#False#True
        self.DoTrai['rf'] = False #True#False #True#True#False#True#True
        self.DoTrai['nn'] = False #True#False #True#True
        self.DoTrain = self.DoTrai['lr'] or self.DoTrai['svcl'] or self.DoTrai['svcg']
        self.DoTrain = self.DoTrain or self.DoTrai['dt'] or self.DoTrai['rf'] or self.DoTrai['nn']

        self.DoShowRes['lr'] = True #Calibrated
        self.DoShowRes['svcl'] = True #Calibrated
        self.DoShowRes['svcg'] = True #Calibrated
        self.DoShowRes['svcl0'] = False #Normal
        self.DoShowRes['svcg0'] = False #Normal
        self.DoShowRes['dt'] = True
        self.DoShowRes['rf'] = True
        self.DoShowRes['nn'] = True

        self.DoPredInpu['lr'] = True#False#True
        self.DoPredInpu['svcl'] = True#False#True#False#True
        self.DoPredInpu['svcg'] = True#False#True
        self.DoPredInpu['svcl0'] = False#True
        self.DoPredInpu['svcg0'] = False#True
        self.DoPredInpu['dt'] = False#True#True
        self.DoPredInpu['rf'] = True#False#True#False#True
        self.DoPredInpu['nn'] = True#False#True

        self.Create_EDAplot = False
        self.DoPlot_ConfusionM = True#False#True
        self.DoWrite_metrics = True#False#True
        self.DoPlot_NNloss = True#False
        self.DoPlot_FeatureImp = False#True

