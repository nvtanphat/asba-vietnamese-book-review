class EarlyStopping:
    def __init__(self,patience=2): self.patience=patience;self.best=float('-inf');self.bad=0
    def step(self,score):
        improved=score>self.best
        if improved:self.best=score;self.bad=0
        else:self.bad+=1
        return improved,self.bad>=self.patience
