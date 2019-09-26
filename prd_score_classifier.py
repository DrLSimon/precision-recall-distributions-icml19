import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from prdataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
from models import *
from inception_torch import InceptionV3

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda:0')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')


def progressbar(iterable, leave=False):
    return tqdm.tqdm(iterable, leave=leave)


def createTrainTestSets(source_folder, target_folder, noise=False):
    transform_test = [transforms.ToTensor()]
    if noise:
        addGaussianNoise = lambda tensor: tensor+torch.randn(tensor.shape)*0.1
        transform_test.append(transforms.Lambda(addGaussianNoise))

    transform_train = transforms.Compose([
        ] + transform_test)

    return SourceTargetDataset(source_folder, target_folder,
                               transform_train=transform_train,
                               transform_test=transforms.ToTensor())


class ClassifierTrainer:
    def __init__(self, dataset, description):
        self.dataset = dataset
        self.totalLoss = np.inf
        self.description = description
        self.__load()

    def __load(self):
        if self.description == 'alex':
            self.features = AlexDiscriminator().eval().to(device)
            self.feat_size = 4096
        elif self.description == 'vgg':
            features = VGGDiscriminator().eval().to(device)
            self.feat_size = 4096
        elif self.description == "inception":
            self.feat_size = dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            features = InceptionV3([block_idx], normalize_input=True)
            self.features = features.eval().to(device)
        else:
            raise ValueError('Unknown classifier')
        self.batch_size = 64
        self.dataset.precomputeFeatures(self.features, self.batch_size, device)

    def initClassifier(self):
        nh=128
        self.classifier  = nn.Sequential(
                nn.Linear(self.feat_size, 1, bias=False),
                )

        self.classifier.to(device).train()
    

    def train(self):
        self.totalLoss=0
        for batch_num, (samples, flips) in enumerate(progressbar(self.train_loader)):
            def closure():
                self.optimizer.zero_grad()
                predictions = self.classifier(samples.to(device))
                loss = self.log_loss(predictions.squeeze(), flips.to(device))
                loss.backward()
                self.totalLoss += float(loss)
                return loss

            self.optimizer.step(closure)


    def test(self):
        self.classifier.eval()
        self.dataset.eval()
        error_I = 0
        error_II = 0
        cnt_I = 0
        cnt_II = 0
        for batch_num, (samples, flips) in enumerate(progressbar(self.train_loader)):
            predictions = self.classifier(samples.to(device))
            predictions = (predictions > 0)
            flips = (flips > 0)
            cnt_I += int((flips.to(device) == 0).sum())
            cnt_II += int((flips.to(device) == 1).sum())
            typeI = (predictions.squeeze() == 1) & (flips.to(device) == 0)
            typeII = (predictions.squeeze() == 0) & (flips.to(device) == 1)
            error_I += int(typeI.sum())
            error_II += int(typeII.sum())
        error_I = float(error_I) / float(cnt_I)
        error_II = float(error_II) / float(cnt_II)
        self.classifier.train()
        self.dataset.train()
        error = 0.5*(error_I + error_II)
        self.scheduler.step(error)
        self.pbar.set_postfix(loss=self.totalLoss, error=f'({error_I:.2}+{error_II:.2})/2={error:.2}', lr=self.optimizer.param_groups[0]['lr'])
        return self.stopper.step(error)

    def run(self, num_epochs, patience):
        early_stopping = (patience >= 1)
        if early_stopping:
            from early_stopping import EarlyStopping
            self.stopper = EarlyStopping(patience=patience)
        self.initClassifier()
        self.dataset.train()
        self.train_loader = DataLoader(self.dataset, self.batch_size, shuffle=True, num_workers=0)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-1, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, cooldown=3, factor=0.5)
        self.log_loss = torch.nn.BCEWithLogitsLoss()
        self.pbar = progressbar(range(num_epochs))
        for ep in self.pbar:
            if early_stopping:
                with torch.no_grad():
                    shouldStop = self.test()
                    if shouldStop:
                        self.pbar.close()
                        break
            self.train()

        return self.classifier

def estimatePRD(classifier, dataset, num_angles, epsilon=1e-10):
    if not (num_angles >= 3 and num_angles <= 1e6):
        raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

    dataset.eval()
    classifier.eval()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # Compute slopes for linearly spaced angles between [0, pi/2]
    angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    toTorch = lambda z: torch.from_numpy(z).unsqueeze(0).to(device)

    with torch.no_grad():
        fValsAndUs = [(float(classifier(Z.to(device))), int(U)) for Z, U in progressbar(test_loader)]
    fVals = [val for val, U in fValsAndUs]
    fVals = [np.min(fVals)-1] + fVals + [np.max(fVals)+1]
    errorRates = []
    for t in fVals:
        fpr=sum([(fOfZ>=t) and U==0 for fOfZ,U in fValsAndUs]) / float(sum([U==0 for fOfZ,U in fValsAndUs]))
        fnr=sum([(fOfZ<t) and U==1 for fOfZ,U in fValsAndUs]) / float(sum([U==1 for fOfZ,U in fValsAndUs]))
        errorRates.append((float(fpr), float(fnr)))
    precision = [] 
    recall = []
    for slope in slopes:
        prec = min([slope*fnr+fpr for fpr,fnr in errorRates])
        precision.append(prec)
        rec =  min([fnr+fpr/slope for fpr,fnr in errorRates])
        recall.append(rec)

    # handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        print(max_val)
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)

    return precision, recall



class EnsembleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.networks=[]

    def append(self, net):
        self.networks.append(net)

    def forward(self, x):
        preds = []
        for net in self.networks:
            preds.append(net(x))
        return torch.median(torch.stack(preds), dim=0)[0]

def computePRD(source_folder, target_folder, num_angles=1001, num_runs=10, num_epochs=10, patience=0):
    precisions = []
    recalls = []
    ensemble = EnsembleClassifier()
    dataset = createTrainTestSets(source_folder, target_folder)
    trainer = ClassifierTrainer(dataset, 'inception')
    for k in progressbar(range(num_runs)):
        classifier = trainer.run(num_epochs, patience)
        ensemble.append(classifier)
    precision, recall = estimatePRD(ensemble, trainer.dataset, num_angles)
    return precision, recall
