import torch
from torch import nn
from torch.nn import functional as F


class LSTNet(nn.Module):
    def __init__(self, P, m, hidR, hidC, hidS, Ck, hw, dropout, output_func, device):
        super(LSTNet, self).__init__()
        self.use_cuda = True
        self.P = P
        self.m = m
        self.hidR = hidR
        self.hidC = hidC
        self.hidS = hidS
        self.Ck = Ck
        self.skip = P - Ck
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = hw
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(dropout);
        
        self.linear=torch.nn.Linear(self.m, 1, device=device)
        
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);

        self.output = None;
        if (output_func == 'sigmoid'):
            self.output = torch.sigmoid;
        if (output_func == 'tanh'):
            self.output = torch.tanh;
 
    def forward(self, x):
        batch_size = x.size(0);
        
        
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        #print(c.shape)
        c = F.relu(self.conv1(c));
        #print(c.shape)
        c = self.dropout(c);
        c = torch.squeeze(c, 3);
        #print(c.shape)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        #print(r.shape)
        with torch.cuda.amp.autocast(): 
         _, r = self.GRU1(r);
         #print(r.shape)
         r = self.dropout(torch.squeeze(r,0));
         #print(r.shape)
        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous(); # c : [6406,100,295] -> s ; [6406,100,294]
            #print(int(-self.pt * self.skip))
            #print(s.shape)
            s = s.view(batch_size, self.hidC, self.pt, self.skip); # s : 
            #print(s.shape)
            s = s.permute(2,0,3,1).contiguous();
            #print(s.shape)
            s = s.view(self.pt, batch_size * self.skip, self.hidC); ##
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
           # print(res.shape)
            
        if (self.output):
            res = self.output(res);
        res = self.linear(res).squeeze(dim=1)
       # print(res)
      #  print(res.shape)
        return res;