#/usr/bin/env

import awkward as ak
from coffea import hist, processor

# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Prepare files from .root skims to a CSV of t event training data')
parser.add_argument('file', metavar='f', type=str)
parser.add_argument('loc',metavar='d', type=str)
args=parser.parse_args()
class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "j1pt":processor.column_accumulator(np.array([])),
                "j1phi":processor.column_accumulator(np.array([])),
                "j1eta":processor.column_accumulator(np.array([])),
                "j1mass":processor.column_accumulator(np.array([])),
                "j2pt":processor.column_accumulator(np.array([])),
                "j2phi":processor.column_accumulator(np.array([])),
                "j2eta":processor.column_accumulator(np.array([])),
                "j2mass":processor.column_accumulator(np.array([])),
                "j3pt":processor.column_accumulator(np.array([])),
                "j3phi":processor.column_accumulator(np.array([])),
                "j3eta":processor.column_accumulator(np.array([])),
                "j3mass":processor.column_accumulator(np.array([])),
                "dR12":processor.column_accumulator(np.array([])),
                "dR13":processor.column_accumulator(np.array([])),
                "dR23":processor.column_accumulator(np.array([])),
                "j1btag":processor.column_accumulator(np.array([])),
                "j2btag":processor.column_accumulator(np.array([])),
                "j3btag":processor.column_accumulator(np.array([])),
                "j1area":processor.column_accumulator(np.array([])),
                "j2area":processor.column_accumulator(np.array([])),
                "j3area":processor.column_accumulator(np.array([])),
                "j12deta":processor.column_accumulator(np.array([])),
                "j23deta":processor.column_accumulator(np.array([])),
                "j13deta":processor.column_accumulator(np.array([])),
                "j12dphi":processor.column_accumulator(np.array([])),
                "j23dphi":processor.column_accumulator(np.array([])),
                "j13dphi":processor.column_accumulator(np.array([])),
                "j1j2mass":processor.column_accumulator(np.array([])),
                "j2j3mass":processor.column_accumulator(np.array([])),
                "j1j3mass":processor.column_accumulator(np.array([])),
                "event":processor.column_accumulator(np.array([])),
                "truth":processor.column_accumulator(np.array([])) })
        print("done")
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self._accumulator.identity()
        jets=events.Jet
        jetSel = (jets.pt>30) & (abs(jets.eta)<2.4)
        tightJet = jets[jetSel]
        bJet = tightJet[tightJet.btagDeepFlavB > 0.642]
        muons = events.Muon
        muonSel = (muons.pt>30) & (abs(muons.eta)<2.4)
        tightMuon = muons[muonSel]
        ele = events.Electron
        eleSel = (ele.pt>35)&(abs(ele.eta)<2.4)
        tightEle = ele[eleSel]
        eventSel = (((ak.num(tightMuon)==1) | (ak.num(tightEle)==1)) &
            (ak.num(tightJet)>= 3) & (ak.num(bJet)>=1)
                   )
        final = events[eventSel]
        
        
        #####GENPART MATCHING ######
        
        genPart = final.GenPart
        tops = genPart[abs(genPart.pdgId)==6]
        #The isLastCopy Flag filters out copy Genparticles:
        tops = tops[tops.hasFlags('isLastCopy')]
        tDecay = tops.distinctChildren
        tDecay = tDecay[tDecay.hasFlags('isLastCopy')]
        t_Events=tDecay[abs(tDecay.pdgId)==5]
        W = tDecay[abs(tDecay.pdgId)==24]
        W = W[W.hasFlags('isLastCopy')]
        WDecay = W.distinctChildren
        WDecay = WDecay[WDecay.hasFlags('isLastCopy')]
        #t_events is the lone bottom, W_events is the -> two jets
        #select the hadronically decaying W
        W_Events=ak.flatten(WDecay[ak.all(abs(WDecay.pdgId)<=8,axis=-1)],axis=3)
        #print(qqb)
        #HadW is mask for Quark deacying W boson
        hadW = ak.num(W_Events,axis=2)==2
        #filters out t_events that have a hadronically decayign W Boson
        hadB = t_Events[hadW]
        hadB = ak.flatten(hadB,axis=2)
        W_quarks = W_Events[hadW]
        W_quarks = ak.flatten(W_quarks,axis=2)
        #concatentating these two arrays make an array of events with the correctly decaying GenParticles.
        qqb = ak.concatenate([hadB,W_quarks],axis=1)
        
        
        #####GEN JET MATCHING ######
        final=final[(ak.count(qqb.pdgId,axis=1)==3)]
        finaljets=final.Jet
        qqb=qqb[(ak.count(qqb.pdgId,axis=1)==3)]
        #Implementing Tight Jet Cuts on Training Data
        finaljetSel=(abs(finaljets.eta)<2.4)&(finaljets.pt>30)
        finalJets=finaljets[finaljetSel]
        #Match Gen part to gen jet
        matchedGenJets=qqb.nearest(final.GenJet)
        #match gen to reco
        matchedJets=matchedGenJets.nearest(finalJets)
    
        ### VALIDATION ###
        test=matchedJets.genJetIdx
        combs=ak.combinations(finalJets,3,replacement=False)
        t1=(combs['0'].genJetIdx==test[:,0])|(combs['0'].genJetIdx==test[:,1])|(combs['0'].genJetIdx==test[:,2])
        t2=(combs['1'].genJetIdx==test[:,0])|(combs['1'].genJetIdx==test[:,1])|(combs['1'].genJetIdx==test[:,2])
        t3=(combs['2'].genJetIdx==test[:,0])|(combs['2'].genJetIdx==test[:,1])|(combs['2'].genJetIdx==test[:,2])
        t=t1&t2&t3
        
        trutharray=ak.flatten(t)
        jetcombos=ak.flatten(combs)
        j1,j2,j3=ak.unzip(jetcombos)
        output["dR12"]+=processor.column_accumulator(ak.to_numpy(j1.delta_r(j2)))
        output["dR13"]+=processor.column_accumulator(ak.to_numpy(j1.delta_r(j3)))
        output["dR23"]+=processor.column_accumulator(ak.to_numpy(j2.delta_r(j3)))
        output["j1btag"]+=processor.column_accumulator(ak.to_numpy(j1.btagCSVV2))
        output["j2btag"]+=processor.column_accumulator(ak.to_numpy(j1.btagCSVV2))
        output["j3btag"]+=processor.column_accumulator(ak.to_numpy(j1.btagCSVV2))
        output["j1area"]+=processor.column_accumulator(ak.to_numpy(j1.area))
        output["j2area"]+=processor.column_accumulator(ak.to_numpy(j2.area))
        output["j3area"]+=processor.column_accumulator(ak.to_numpy(j3.area))
        output["j12deta"]+=processor.column_accumulator(ak.to_numpy(j1.eta-j2.eta))
        output["j23deta"]+=processor.column_accumulator(ak.to_numpy(j2.eta-j3.eta))
        output["j13deta"]+=processor.column_accumulator(ak.to_numpy(j1.eta-j3.eta))
        output["j12dphi"]+=processor.column_accumulator(ak.to_numpy(j1.phi-j2.phi))
        output["j23dphi"]+=processor.column_accumulator(ak.to_numpy(j2.phi-j3.phi))
        output["j13dphi"]+=processor.column_accumulator(ak.to_numpy(j1.phi-j3.phi))
        output["j1j2mass"]+=processor.column_accumulator(ak.to_numpy(j1.mass+j2.mass))
        output["j2j3mass"]+=processor.column_accumulator(ak.to_numpy(j2.mass+j3.mass))
        output["j1j3mass"]+=processor.column_accumulator(ak.to_numpy(j1.mass+j3.mass))
        output["j1pt"]+=processor.column_accumulator(ak.to_numpy(j1.pt))
        output["j1phi"]+=processor.column_accumulator(ak.to_numpy(j1.phi))
        output["j1eta"]+=processor.column_accumulator(ak.to_numpy(abs(j1.eta)))
        output["j1mass"]+=processor.column_accumulator(ak.to_numpy(j1.mass))
        output["j2pt"]+=processor.column_accumulator(ak.to_numpy(j2.pt))
        output["j2phi"]+=processor.column_accumulator(ak.to_numpy(j2.phi))
        output["j2eta"]+=processor.column_accumulator(ak.to_numpy(abs(j2.eta)))
        output["j2mass"]+=processor.column_accumulator(ak.to_numpy(j2.mass))
        output["j3pt"]+=processor.column_accumulator(ak.to_numpy(j3.pt))
        output["j3phi"]+=processor.column_accumulator(ak.to_numpy(j3.phi))
        output["j3eta"]+=processor.column_accumulator(ak.to_numpy(abs(j3.eta)))
        output["j3mass"]+=processor.column_accumulator(ak.to_numpy(j3.mass))
        output["event"]+=processor.column_accumulator(ak.to_numpy(ak.flatten(ak.broadcast_arrays(final.event,combs['0'].pt)[0])))
        output["truth"]+=processor.column_accumulator(ak.to_numpy(trutharray).astype(int))
        
        return output

    def postprocess(self, accumulator):
            return accumulator 
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import pandas as pd
class HackSchema(NanoAODSchema):
    def __init__(self, base_form):
        base_form["contents"].pop("Muon_fsrPhotonIdx", None)
        base_form["contents"].pop("Electron_photonIdx", None)
        super().__init__(base_form)
print(args.file)
f=args.file
files = {"TTBAR":[args.file]}
result = processor.run_uproot_job(
    files,
    treename="Events",
    processor_instance=MyProcessor(),
    executor=processor.iterative_executor,
    executor_args={'schema':HackSchema},
    chunksize=10000
)

l=args.loc
keys=result.keys()
finaldict={}
for key in keys:
    finaldict[key]=result[key].value
df=ak.to_pandas(ak.zip(finaldict))
df.to_csv("%s/%sDatasetWithTruths.csv"%(l,f[0:-5]))