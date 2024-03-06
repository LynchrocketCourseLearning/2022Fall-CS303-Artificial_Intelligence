import torch
import torch.nn as nn
from functorch import vmap
from functorch import grad
from typing import Tuple
from project3.my_network import *
from project3.src import *
import time
import os

class Agent:
    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """
        # TODO: prepare your agent here
        self.classifier = MyClassifier()
        model_path = os.path.join(os.path.dirname(__file__), 'myModel4.pth')
        self.classifier.load_state_dict(torch.load(model_path))


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
        target_cls = torch.stack([nn.Softmax(dim=0)(self.classifier(features)).argmax(0) for features in target_features])    
        
        ctps_inter_list = [torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.]) for _ in range(20)]
        ctps_inter_list = map(lambda x: x.requires_grad_(True), ctps_inter_list)
        
        my_lr = 0.260
        optimizer_list = [torch.optim.Adam([x], lr=my_lr, maximize=True) for x in ctps_inter_list]
        
        for epoch in range(16):
            optimizer_list = map(lambda x: x.zero_grad(), optimizer_list)
            gra_score_list = [self.evaluate_modified(compute_traj(x), target_pos, class_scores[target_cls], RADIUS) for x in ctps_inter_list]
            gra_score_list = map(lambda x: x.backward(), gra_score_list)
            optimizer_list = map(lambda x: x.step(), optimizer_list)
            ed = time.time()
            if ed - st >= 0.25:
                break
        real_score_list = [evaluate(compute_traj(x), target_pos, class_scores[target_cls], RADIUS) for x in ctps_inter_list]
        return ctps_inter_list[real_score_list.index(max(real_score_list))]
 
def evaluate_modified (
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
            
def compute_evaluate(ctps_inter, target_pos, class_scores, target_cls):
    return evaluate_modified(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)

def compute_grad(ctps_inter, target_pos, class_scores, target_cls):
    return grad(compute_evaluate)(ctps_inter, target_pos, class_scores, target_cls)


