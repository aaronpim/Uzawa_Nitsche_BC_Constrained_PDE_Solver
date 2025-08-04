import numpy as np 
import math, torch, generateData, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
from areaVolume import areaVolume
import argparse

# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        x_ori = x
        r = torch.sum(x.square(), axis =1 ).unsqueeze(1)
        f = exact(1, x_ori)
        x = F.silu(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x_temp = F.silu(layer(x))
            x = x_temp+x
        u = self.linearOut(x)
        output = (1 - r) * u + r * f

        return output

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def preTrain(model,device,params,preOptimizer,preScheduler,fun):
    model.train()
    file = open(f"Error_Poisson_Hard_dim={params["d"]}_SEED={params["SEED"]}.txt","w")

    for step in range(params["preStep"]):
        # The volume integral
        data = torch.from_numpy(generateData.sampleFromDiskD(params["radius"],params["bodyBatch"],params["d"])).float().to(device)

        output = model(data)

        target = fun(params["radius"],data)

        loss = output-target
        loss = torch.mean(loss*loss)

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                ref = exact(params["radius"],data)
                error = errorFun(output,ref,params)
                # print("Loss at Step %s is %s."%(step+1,loss.item()))
                print("Error at Step %s is %s."%(step+1,error))
            file.write(str(step+1)+" "+str(error)+"\n")

        model.zero_grad()
        loss.backward()

        # Update the weights.
        preOptimizer.step()

def train(model,device,params,optimizer,scheduler):
    model.train()

    data1 = torch.from_numpy(generateData.sampleFromDiskD(params["radius"],params["bodyBatch"],params["d"])).float().to(device)
    data1.requires_grad = True

    for step in range(params["trainStep"]-params["preStep"]):
        output1 = model(data1)

        model.zero_grad()

        dfdx = torch.autograd.grad(output1,data1,grad_outputs=torch.ones_like(output1),retain_graph=True,create_graph=True,only_inputs=True)[0]
        # Loss function 1
        fTerm = ffun(data1).to(device)
        loss1 = torch.mean(0.5*torch.sum(dfdx*dfdx,1).unsqueeze(1)-fTerm*output1)

        loss = loss1

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                target = exact(params["radius"],data1)
                error = errorFun(output1,target,params)
                print("Error at Step %s is %s."%(step+params["preStep"]+1,error))
            file = open(f"Error_Poisson_Hard_dim={params["d"]}_SEED={params["SEED"]}.txt","a")
            file.write(str(step+params["preStep"]+1)+" "+str(error)+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:
            data1 = torch.from_numpy(generateData.sampleFromDiskD(params["radius"],params["bodyBatch"],params["d"])).float().to(device)
            data1.requires_grad = True
            data2 = torch.from_numpy(generateData.sampleFromSurfaceD(params["radius"],params["bdryBatch"],params["d"])).float().to(device)

        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        loss.backward()

        optimizer.step()
        scheduler.step()

def errorFun(output,target,params):
    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref   

def test(model,device,params):
    numQuad = params["numQuad"]

    data = torch.from_numpy(generateData.sampleFromDiskD(1,numQuad,params["d"])).float().to(device)
    output = model(data)
    target = exact(params["radius"],data).to(device)

    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref

def ffun(data):
    return 0.0*torch.ones([data.shape[0],1],dtype=torch.float)

def exact(r,data):
    s = data.shape
    N = s[1]
    output = 0
    if N % 2 == 0:
        for i in range(0,N,2):
            output = output + data[:,i]*data[:,i+1]
    else:
        for i in range(0,N-1,2):
            output = output + data[:,i]*data[:,i+1]
        output = output + data[:,N-1]
    return output.unsqueeze(1)

def rough(r,data):
    output = torch.zeros(data.shape[0],dtype=torch.float)
    return output.unsqueeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) # if p.requires_grad

def main(seed = 1, dimension_N = 8):
    # Parameters

    params = dict()
    params["SEED"] = seed
    params["radius"] = 1
    params["d"] = dimension_N # dimension
    params["dd"] = 1 # Scalar field
    params["bodyBatch"] = 1024 # Batch size
    params["bdryBatch"] = 2048 # Batch size for the boundary integral
    params["lr"] = 0.001 # Learning rate
    params["preLr"] = params["lr"] # Learning rate (Pre-training)
    params["width"] = 40 # Width of layers
    params["depth"] = 5 # Depth of the network: depth+2
    params["numQuad"] = 40000 # Number of quadrature points for testing
    params["trainStep"] = 50000
    params["preStep"] = 0
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["area"] = areaVolume(params["radius"],params["d"])
    params["step_size"] = 5000
    params["milestone"] = [5000,10000,20000,35000,48000]
    params["gamma"] = 1.0
    params["decay"] = 0.0001

    torch.manual_seed(params["SEED"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    startTime = time.time()
    model = RitzNet(params).to(device)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])

    startTime = time.time()
    preTrain(model,device,params,preOptimizer,None,rough)
    train(model,device,params,optimizer,scheduler)
    file = open(f"TIME_Poisson_Hard.txt","a")
    file.write(str(params["d"])+" "+str(params["trainStep"])+" "+str(time.time()-startTime)+"\n")

    model.eval()
    testError = test(model,device,params)
    print("The test error (of the last model) is %s."%testError)
    print("The number of parameters is %s,"%count_parameters(model))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PINN Training')
    parser.add_argument('--SEED', type=int, default=1)
    parser.add_argument('--dim', type=int, default=8) # dimension of the problem.
    args = parser.parse_args()
    main( args.SEED, args.dim )
