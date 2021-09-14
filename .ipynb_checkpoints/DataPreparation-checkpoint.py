#/usr/bin/env

# Top Quark ML data preparation
# By: Zach Shelton
# Purpose: Use .root files, usually skims and converts them to csv's of sets of combinations with kinematics data for 
# Boosted Decision Tree
from __future__ import print_function, division
import numpy as np
from coffea.nanoevents import NanoAODSchema,NanoEventsFactory
import coffea
from uproot3_methods import TLorentzVectorArray
import uproot3_methods
import numpy as np
import coffea.hist as hist
import uproot
import awkward1 as ak
class HackSchema(NanoAODSchecoffema):
    def __init__(self, base_form):
        base_form["contents"].pop("Muon_fsrPhotonIdx", None)
        base_form["contents"].pop("Electron_photonIdx", None)
        super().__init__(base_form)
import argparse
parser = argparse.ArgumentParser(description='Prepare files from .root skims to a CSV of t event training data')
parser.add_argument('file', metavar='f', type=str)
parser.add_argument('loc',metavar='d', type=str)
args=parser.parse_args()
from pprint import pprint
##InitialDataCuts

events = NanoEventsFactory.from_root(args.file,schemaclass=HackSchema).events()

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

print("initial cuts made")
#Finding Gen Part Events
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


#HadW is mask for Quark deacying W boson
hadW = ak.num(W_Events,axis=2)==2
#filters out t_events that have a hadronically decayign W Boson
hadB = t_Events[hadW]
hadB = ak.flatten(hadB,axis=2)

W_quarks = W_Events[hadW]
W_quarks = ak.flatten(W_quarks,axis=2)
#concatentating these two arrays make an array of events with the correctly decaying GenParticles.
qqb = ak.concatenate([hadB,W_quarks],axis=1)

print("qqb Genparts matched")

#Filtering Out events with extra tops
final=final[(ak.count(qqb.pdgId,axis=1)==3)]
finaljets=final.Jet
qqb=qqb[(ak.count(qqb.pdgId,axis=1)==3)]
#Implementing Tight Jet Cuts on Training Data
finaljetSel=(abs(finaljets.eta)<2.4)&(finaljets.pt>30)
finalJets=finaljets[finaljetSel]

#Use nearest to match Jets
matchedGenJets=qqb.nearest(final.GenJet)
matchedJets=matchedGenJets.nearest(finalJets)

print("matched genpart to genjet and finally to reco jets")

#Assigning True false to sets of 3 jets
test=matchedJets.genJetIdx
combs=ak.combinations(finalJets,3,replacement=False)
t1=(combs['0'].genJetIdx==test[:,0])|(combs['0'].genJetIdx==test[:,1])|(combs['0'].genJetIdx==test[:,2])
t2=(combs['1'].genJetIdx==test[:,0])|(combs['1'].genJetIdx==test[:,1])|(combs['1'].genJetIdx==test[:,2])
t3=(combs['2'].genJetIdx==test[:,0])|(combs['2'].genJetIdx==test[:,1])|(combs['2'].genJetIdx==test[:,2])
t=t1&t2&t3
trutharray=ak.flatten(t)

print("matching a validation array for every combo of 3 jets")

#Zipping into CSV for training
jetcombos=ak.flatten(combs)
j1,j2,j3=ak.unzip(jetcombos)
dR1_2=j1.delta_r(j2)
dR1_3=j1.delta_r(j3)
dR2_3=j2.delta_r(j3)
j1b_tag=j1.btagCSVV2
j2b_tag=j1.btagCSVV2
j3b_tag=j1.btagCSVV2
j1area=j1.area
j2area=j2.area
j3area=j3.area
j12deta=j1.eta-j2.eta
j23deta=j2.eta-j3.eta
j13deta=j1.eta-j3.eta
j12dphi=j1.phi-j2.phi
j23dphi=j2.phi-j3.phi
j13dphi=j1.phi-j3.phi
j1j2mass=j1.mass+j2.mass
j2j3mass=j2.mass+j3.mass
j1j3mass=j1.mass+j3.mass
j1pt=j1.pt
j1phi=j1.phi
j1eta=abs(j1.eta)
j1mass=j1.mass
j2pt=j2.pt
j2phi=j2.phi
j2eta=abs(j2.eta)
j2mass=j2.mass
j3pt=j3.pt
j3phi=j3.phi
j3eta=abs(j3.eta)
j3mass=j3.mass
event=ak.flatten(ak.broadcast_arrays(final.event,combs['0'].pt)[0])
processedMLdata=ak.zip({"j1pt":j1.pt,"j1phi":j1.phi,"j1eta":j1.eta,"j1mass":j1.mass,
                         "j2pt":j2.pt,"j2phi":j2.phi,"j2eta":j2.eta,"j2mass":j2.mass,
                          "j3pt":j3.pt,"j3phi":j3.phi,"j3eta":j3.eta,"j3mass":j3.mass,
                        "dR12":dR1_2,"dR13":dR1_3,"dR23":dR2_3,
                       "j1btag":j1b_tag,"j2btag":j2b_tag,"j3btag":j3b_tag,
                       "j1area":j1area,"j2area":j2area,"j3area":j3area,
                        "j12deta":j12deta,"j23deta":j23deta,"j13deta":j13deta,
                       "j12dphi":j12dphi,"j23dphi":j23dphi,"j13dphi":j13dphi,
                       "j1j2mass":j1j2mass,"j2j3mass":j2j3mass,"j1j3mass":j1j3mass,"event":event})

print("Zipping into awkward and then pandas CSV")
df=ak.to_pandas(processedMLdata)
df.to_csv("%s/%sData.csv"%(args.loc,args.file),index=False)
vf=ak.to_pandas(trutharray)
vf.to_csv("%s/%sValidation.csv"%(args.loc,args.file),index=False)