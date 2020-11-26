import torch

def _L3(theta):
    """

    :param theta: interation score shape(N*(N-1),2)
    :return:loss
    """
    tail=torch.log(torch.exp(theta[:,0])+torch.exp(theta[:,1]))
    return torch.sum(-theta[:,0]+tail)

def L3(theta,boxNum):
    l=boxNum.shape[0]
    head=0
    loss=0.
    for i in range(l):
        n=boxNum[i]
        t=theta[head:head+n*(n-1),1]
        loss+=torch.sum(t)/t.shape[0]
        head+=n*(n-1)
    return loss/l


def L4(mat,groundMat,boxNum):
    loss = torch.sum((mat - groundMat).pow(2)) / mat.shape[0]
    return loss

def inter_mse(zp,z,boxn):
    return L4(zp,z,boxn)


if __name__=="__main__":
    zp = torch.Tensor([1,1,1,1])
    z = torch.Tensor([1,0,0,0])
    boxn=torch.Tensor([2,2]).long()
    print(inter_mse(zp,z,boxn))
