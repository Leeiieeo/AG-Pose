import torch

def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=2))
    return dis

def ChamferDis(p1, p2):
    '''
    p1: b*n1*3
    p2: b*n2*3
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    dis = 0.5*dis1.mean(1) + 0.5*dis2.mean(1)
    return dis.mean()

def ChamferDis_wo_Batch(p1, p2):
    """
    Args:
        p1: (n1, 3)
        p2: (n2, 3)
    """
    dis = torch.norm(p1.unsqueeze(1) - p2.unsqueeze(0), dim=2) # (n1, n2)
    dis1 = torch.min(dis, 1)[0] # (n1, )
    dis2 = torch.min(dis, 0)[0] # (n2, )
    dis = 0.5*dis1.mean() + 0.5*dis2.mean()
    return dis

def PoseDis(r1, t1, s1, r2, t2, s2):
    '''
    r1, r2: b*3*3
    t1, t2: b*3
    s1, s2: b*3
    '''
    dis_r = torch.mean(torch.norm(r1 - r2, dim=1))
    dis_t = torch.mean(torch.norm(t1 - t2, dim=1))
    dis_s = torch.mean(torch.norm(s1 - s2, dim=1))

    return dis_r + dis_t + dis_s

def UniChamferDis(p1, p2):
    '''
    p1: b, n1, 3
    p2: b, n2, 3
    '''
    # (b, n1, n2)
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis = torch.min(dis, 2)[0]

    return dis.mean()
