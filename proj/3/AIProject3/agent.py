import torch
import torch.nn as nn
from typing import Tuple
from src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate, compute_traj
import time

class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """

        # TODO: prepare your agent here
        self.classifier = torch.load('myModel2.pth', map_location='cpu')

    def evaluate(
        self,
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        cdist = torch.cdist(target_pos, traj) # see https://pytorch.org/docs/stable/generated/torch.cdist.html
        d = cdist.min(-1).values
        hit = (d < radius)
        d[hit] = 1
        # d[~hit] = radius / d[~hit]
        # d[~hit] = 1e-20 / d[~hit]
        d[~hit] = 1 - (torch.exp(d[~hit])-torch.exp(-d[~hit])) / (torch.exp(d[~hit])+torch.exp(-d[~hit]))
        # d[~hit] = 1 - 1 / (1+torch.exp(-d[~hit]))
        value = torch.sum(d * target_scores, dim=-1)
        return value

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile. 

        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        assert len(target_pos) == len(target_features)

        # TODO: compute the firing speed and angle that would give the best score.
        st = time.time()
        flag = False
        target_cls = torch.stack([nn.Softmax(dim=0)(self.classifier(features)).argmax(0) for features in target_features])    
        
        ctps_inter_ret = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
        real_score_ret = evaluate(compute_traj(ctps_inter_ret), target_pos, class_scores[target_cls], RADIUS)
        my_lr = 0.260
        # my_lr = 0.99
        for p in range(11):
            # ctps_inter = ctps_inter_ret
            ctps_inter = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
            ctps_inter.requires_grad = True
            
            optimizer = torch.optim.Adam([ctps_inter], lr=my_lr, maximize=True)
            # optimizer = torch.optim.SGD([ctps_inter], lr=my_lr, momentum=0.5, maximize=True)
            # optimizer.zero_grad()
            # gra_score = self.evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            # gra_score.backward()
            # optimizer.step()
            
            # now_score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            # if now_score < real_score_ret:
            #     ctps_inter = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
            #     ctps_inter.requires_grad = True
            #     optimizer = torch.optim.Adam([ctps_inter], lr=my_lr, maximize=True)
                # optimizer = torch.optim.SGD([ctps_inter], lr=my_lr, momentum=0.5, maximize=True)
                
            for it in range(12):
                optimizer.zero_grad()
                gra_score = self.evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
                # real_score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
                # print(it, real_score.item())
                gra_score.backward()
                optimizer.step()
                # ctps_inter.data = ctps_inter.data + my_lr * ctps_inter.grad / torch.norm(ctps_inter.grad)
                ed = time.time()
                if ed - st >= 0.29:
                    flag = True
                    break
                
            real_score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            if real_score > real_score_ret:
                ctps_inter_ret = ctps_inter
                real_score_ret = real_score
            if flag:
                break
            
        return ctps_inter_ret
