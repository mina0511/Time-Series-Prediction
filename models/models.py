import torch
from torch import nn


class EncoderLSTM(nn.Module):
    def __init__(self, n, m): # m: dimension of Encoder hidden state
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = n, hidden_size = m, batch_first = True)
        self.initial_state = None
    
    def forward(self, x_t, training = False):
        _, (h_t, s_t) = self.lstm(x_t, self.initial_state) # (1, -1 x m), (1, -1, m)
        self.initial_state = (h_t, s_t)
        return h_t, s_t

    def reset_state(self, h_0, s_0):
        self.initial_state = (h_0, s_0)


class InputAttention(nn.Module):
    def __init__(self, T, n, m):
        super(InputAttention, self).__init__()
        self.v_e = nn.Linear(T, 1, bias = False)
        self.W_e = nn.Linear(2*m, T, bias = False)
        self.U_e = nn.Linear(T, T, bias = False)
        self.softmax = nn.Softmax(dim = 2)
        self.n = n
    
    def forward(self, h_t, s_t, data): # actually ( h_(t-1), s_(t-1), ( x(1), ..., x(n) ) )
        h_t = h_t.permute(1, 0, 2) # (-1 x 1 x m)
        s_t = s_t.permute(1, 0, 2) # (-1 x 1 x m)
        query = torch.concat([h_t, s_t], dim = 2) # (-1 x 1 x 2m)
        query = torch.concat([query] * self.n, dim = 1) # (-1 x n x 2m)

        alpha_t = torch.tanh(self.W_e(query) + self.U_e(data.permute(0, 2, 1))) # (-1 x n x T)
        alpha_t = self.v_e(alpha_t).permute(0, 2, 1) # (-1 x n x 1)
        alpha_t = self.softmax(alpha_t) # (-1 x 1 x n)

        return alpha_t # (-1 x 1 x n)


class Encoder(nn.Module):
    def __init__(self, T, n, m):
        super(Encoder, self).__init__()
        self.T = T
        self.input_attention_score = InputAttention(T, n, m)
        self.lstm = EncoderLSTM(n, m)
    

    def forward(self, data, h_0, s_0):
        self.lstm.reset_state(h_0, s_0)
        h_t, s_t = h_0, s_0 # (1 x -1 x n)
        
        H = []
        attention_scores = []
        for t in range(self.T):
            # finding x_t_tilde
            x_t = data[:, [t], :] # (-1 x 1 x n)
            alpha_t = self.input_attention_score(h_t, s_t, data) # (-1 x 1 x n)
            # h_(t-1), s_(t-1), [x(1) ... x(n)] -> alpha_(t)

            attention_scores.append(alpha_t)
            x_t_tilde = x_t.mul(alpha_t) # (-1 x 1 x n)

            # update hidden state
            h_t, s_t = self.lstm(x_t_tilde) # (1 x -1 x m)
            H.append(h_t.permute(1, 0, 2)) # (-1 x 1 x m)

        attention_scores = torch.cat(attention_scores, dim=1) # (-1 x T x n)
        H = torch.cat(H, dim=1) # (-1 x T x m)

        return H, attention_scores


class Model1(nn.Module):
    def __init__(self, T, n, m, cnn_kernel_height=5, cnn_hidden_size=40, skip_hidden_size=24, skip=10):
        super(Model1, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state
        # p: dimension of Deocder hidden state

        self.m = m; self.T = T

        # Part of LSTNet
        self.cnn_kernel_height = cnn_kernel_height
        self.cnn_kernel_width = n
        self.T_modified = self.T - self.cnn_kernel_height + 1 # T'
        self.skip = skip
        self.skip_hidden_size = skip_hidden_size
        self.pt = int(self.T_modified / self.skip)
        self.cnn_hidden_size = cnn_hidden_size
        self.layer_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.cnn_hidden_size,
                kernel_size=(self.cnn_kernel_height, self.cnn_kernel_width),
                stride=1, padding='valid'
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) # (-1 x 1 x T x n) -> (-1 x `cnn_hidden_size` x T' x 1)
        if self.skip > 0:
            self.gru_skip = nn.GRU(self.cnn_hidden_size, self.skip_hidden_size)

        # Part of DARNN
        self.encoder = Encoder(T = self.T_modified, n = self.cnn_hidden_size, m = m)

        # Part of Result
        self.layer_cnn_beta1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(self.T_modified, 1),
                stride=1, padding='valid'
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) # (-1 x 1 x T' x m) -> (-1 x 1 x 1 x m)

        self.linear_output = nn.Sequential(
            nn.Linear(m + self.skip*self.skip_hidden_size, m),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(m, 1)
        ) # (-1 x m + `skip`*`skip_hidden_size`) -> (-1 x 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):
        # enc: (-1 x T x n)
        # dec: (-1 x (T-1) x 1)
        batch_size = enc_data.shape[0]
        self.device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.m, device = self.device)
        s0 = torch.zeros(1, batch_size, self.m, device = self.device)

        # 1. feature-wise CNN
        enc_data = enc_data.view(-1, 1, *enc_data.shape[1:]) # (-1 x 1 x T x n)
        enc_data = self.layer_cnn(enc_data) # (-1 x `cnn_hidden_size` x T' x 1)
        enc_data = enc_data.squeeze(dim=3).permute(0, 2, 1) # (-1 x T' x `cnn_hidden_size`)

        # 2. Encoder
        H, self.attention_scores_ = self.encoder(data = enc_data, h_0 = h0, s_0 = s0) # (-1 x T' x m)
        H = H.view(-1, 1, *H.shape[1:]) # (-1 x 1 x T' x m)
        output = self.layer_cnn_beta1(H).squeeze() # (-1 x m)

        # 3. LSTNet: skip-RNN
        if self.skip > 0:
            enc_data_skipped = enc_data[:, -self.pt*self.skip:, :].permute(0, 2, 1).contiguous() # (-1 x m x p)
            enc_data_skipped = enc_data_skipped.view(batch_size, self.cnn_hidden_size, self.pt, self.skip) # (-1 x T x `pt` x `skip`)
            enc_data_skipped = enc_data_skipped.permute(2, 0, 3, 1).contiguous() # (`pt` x -1 x `skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(self.pt, batch_size * self.skip, self.cnn_hidden_size) # (`pt` x -1*skip x `cnn_hidden_size`)
            _, h_skip = self.gru_skip(enc_data_skipped) # (1 x -1*`skip` x `skip_hidden_size`)

            h_skip = h_skip.view(batch_size, -1, self.skip * self.skip_hidden_size) # (-1 x 1 x `skip` * `skip_hidden_size`)
            h_skip = self.dropout(h_skip).squeeze(dim=1) # (-1 x `skip`*`skip_hidden_size`)
            output = torch.cat([output, h_skip], dim=1) # (-1 x m+`skip`*`skip_hidden_size`)

        output = self.linear_output(output).squeeze(dim=1) # (-1)

        return output


class Model2(nn.Module):
    def __init__(self, T, n, m, skip_hidden_size, T_modified, skip=10):
        super(Model2, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state

        self.m = m; self.T = T

        # Part of CNN
        self.cnn_kernel_height = T
        self.cnn_kernel_width = 1
        self.T_modified = T_modified # self.T - self.cnn_kernel_height + 1 = T'
        self.skip = skip
        self.skip_hidden_size = skip_hidden_size
        self.pt = int(self.T_modified / self.skip)
        self.cnn_hidden_size = m
        self.layer_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=T_modified,
                kernel_size=(self.cnn_kernel_height, self.cnn_kernel_width),
                stride=1, padding='valid'
            ), # (-1 x 1 x T x m) -> (-1 x T' x 1 x m)  ||  NO > T' = T - `cnn_kernel_height` + 1
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Part of DARNN
        self.encoder = Encoder(T=self.T, n=n, m=m) # (-1 x T x m)
        
        # Part of LSTNet
        if self.skip > 0:
            self.gru_skip = nn.GRU(self.cnn_hidden_size, self.skip_hidden_size)

        # Part of Result
        self.layer_cnn_beta1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(self.T_modified, 1),
                stride=1, padding='valid'
            ),
            # (-1 x 1 x T x m) -> (-1 x 1 x 1 x m)
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.linear_beta1 = nn.Sequential(
            nn.Linear(self.T_modified, 1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.linear_output = nn.Linear(m + self.skip*self.skip_hidden_size, 1) # (-1 x 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):
        # enc: (-1 x T x n)
        batch_size = enc_data.shape[0]
        self.device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.m, device = self.device)
        s0 = torch.zeros(1, batch_size, self.m, device = self.device)

        # 1. Encoder
        H, self.attention_scores_ = self.encoder(enc_data, h_0 = h0, s_0 = s0) # (-1 x T x m)
        H = H.view(-1, 1, self.T, self.m) # (-1 x 1 x T x m)

        # 2. Temporal CNN
        H = self.layer_cnn(H).squeeze(dim=2) # (-1 x T' x m)
        output = self.linear_beta1(H.permute(0, 2, 1)) # (-1 x m x 1)
        output = output.squeeze(dim=2) # (-1 x m)

        # skip-RNN
        if self.skip > 0:
            enc_data_skipped = H[:, -self.pt*self.skip:, :].permute(0, 2, 1).contiguous() # (-1 x `pt`*`skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(batch_size, self.cnn_hidden_size, self.pt, self.skip) # (-1 x `cnn_hidden_size` x `pt` x `skip`)
            enc_data_skipped = enc_data_skipped.permute(2, 0, 3, 1).contiguous() # (`pt` x -1 x `skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(self.pt, batch_size * self.skip, self.cnn_hidden_size) # (`pt` x -1*skip x `cnn_hidden_size`)
            _, h_skip = self.gru_skip(enc_data_skipped) # (1 x -1*`skip` x `skip_hidden_size`)

            h_skip = h_skip.view(batch_size, -1, self.skip * self.skip_hidden_size) # (-1 x 1 x `skip` * `skip_hidden_size`)
            h_skip = self.dropout(h_skip).squeeze(dim=1) # (-1 x `skip`*`skip_hidden_size`)
            output = torch.cat([output, h_skip], dim=1) # (-1 x m+`skip`*`skip_hidden_size`)
        
        output = self.linear_output(output).squeeze(dim=1) # (-1)
        # please please please please please please please please please please be better

        return output


class Model2_2(nn.Module):
    def __init__(self, T, n, skip_hidden_size, T_modified, skip=10):
        super(Model2_2, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state

        self.T = T

        # Part of CNN
        self.cnn_kernel_height = T
        self.cnn_kernel_width = 1
        self.T_modified = T_modified # self.T - self.cnn_kernel_height + 1 = T'
        self.skip = skip
        self.skip_hidden_size = skip_hidden_size
        self.pt = int(self.T_modified / self.skip)
        self.cnn_hidden_size = n
        self.layer_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=T_modified,
                kernel_size=(self.cnn_kernel_height, self.cnn_kernel_width),
                stride=1, padding='valid'
            ), # (-1 x 1 x T x m) -> (-1 x T' x 1 x m)  ||  NO > T' = T - `cnn_kernel_height` + 1
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Part of LSTNet
        if self.skip > 0:
            self.gru_skip = nn.GRU(self.cnn_hidden_size, self.skip_hidden_size)

        self.linear_beta1 = nn.Sequential(
            nn.Linear(self.T_modified, 1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.linear_output = nn.Linear(n + self.skip*self.skip_hidden_size, 1) # (-1 x 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):  # (-1 x T x n)
        # enc: (-1 x T x n)
        batch_size = enc_data.shape[0]

        # 2. Temporal CNN
        enc_data = enc_data.unsqueeze(dim=1)  # (-1 x 1 x T x n)
        enc_data = self.layer_cnn(enc_data).squeeze(dim=2)  # (-1 x T' x n)
        output = self.linear_beta1(enc_data.permute(0, 2, 1)) # (-1 x n x 1)
        output = output.squeeze(dim=2) # (-1 x n)

        # skip-RNN
        if self.skip > 0:
            enc_data_skipped = enc_data[:, -self.pt*self.skip:, :].permute(0, 2, 1).contiguous() # (-1 x `pt`*`skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(batch_size, self.cnn_hidden_size, self.pt, self.skip) # (-1 x `cnn_hidden_size` x `pt` x `skip`)
            enc_data_skipped = enc_data_skipped.permute(2, 0, 3, 1).contiguous() # (`pt` x -1 x `skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(self.pt, batch_size * self.skip, self.cnn_hidden_size) # (`pt` x -1*skip x `cnn_hidden_size`)
            _, h_skip = self.gru_skip(enc_data_skipped) # (1 x -1*`skip` x `skip_hidden_size`)

            h_skip = h_skip.view(batch_size, -1, self.skip * self.skip_hidden_size) # (-1 x 1 x `skip` * `skip_hidden_size`)
            h_skip = self.dropout(h_skip).squeeze(dim=1) # (-1 x `skip`*`skip_hidden_size`)
            output = torch.cat([output, h_skip], dim=1) # (-1 x n+`skip`*`skip_hidden_size`)
        
        output = self.linear_output(output).squeeze(dim=1) # (-1)
        # please please please please please please please please please please be better

        return output


class Model3(nn.Module):
    def __init__(self, T, n, m, skip_hidden_size, T_modified, skip=10):
        super(Model3, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state

        self.m = m; self.T = T

        # Part of CNN
        self.cnn_kernel_height = T
        self.cnn_kernel_width = 1
        self.T_modified = T_modified # self.T - self.cnn_kernel_height + 1 = T'
        self.skip = skip
        self.skip_hidden_size = skip_hidden_size
        self.pt = int(self.T_modified / self.skip)
        self.cnn_hidden_size = n
        self.layer_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=self.T_modified,
                kernel_size=(self.cnn_kernel_height, self.cnn_kernel_width),
                stride=1, padding='valid'
            ), # (-1 x 1 x T x n) -> (-1 x T' x 1 x n)  ||  NO > T' = T - `cnn_kernel_height` + 1
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Part of DARNN
        self.encoder = Encoder(T=self.T_modified, n=n, m=m) # (-1 x T x m)
        
        # Part of LSTNet
        if self.skip > 0:
            self.gru_skip = nn.GRU(self.cnn_hidden_size, self.skip_hidden_size)

        # Part of Result
        self.layer_linear_beta1 = nn.Sequential(
            nn.Linear(self.T_modified, 1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) # (-1 x m x T') -> (-1 x m x 1)

        self.linear_output = nn.Linear(m + self.skip*self.skip_hidden_size, 1) # (-1 x 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):
        # enc: (-1 x T x n)
        batch_size = enc_data.shape[0]
        self.device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.m, device = self.device)
        s0 = torch.zeros(1, batch_size, self.m, device = self.device)

        # 1. Temporal CNN
        enc_data = enc_data.view(-1, 1, *enc_data.shape[1:])
        enc_data = self.layer_cnn(enc_data) # (-1 x T' x 1 x n)
        enc_data = enc_data.squeeze(dim=2) # (-1 x T' x n)

        # 2. Encoder
        H, self.attention_scores_ = self.encoder(enc_data, h_0 = h0, s_0 = s0) # (-1 x T' x m)
        H = H.permute(0, 2, 1) # (-1 x m x T')
        output = self.layer_linear_beta1(H) # (-1 x m x 1)
        output = output.squeeze(dim=2) # (-1 x m)

        # skip-RNN
        if self.skip > 0:
            enc_data_skipped = enc_data[:, -self.pt*self.skip:, :].permute(0, 2, 1).contiguous() # (-1 x `pt`*`skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(batch_size, self.cnn_hidden_size, self.pt, self.skip) # (-1 x `cnn_hidden_size` x `pt` x `skip`)
            enc_data_skipped = enc_data_skipped.permute(2, 0, 3, 1).contiguous() # (`pt` x -1 x `skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(self.pt, batch_size * self.skip, self.cnn_hidden_size) # (`pt` x -1*skip x `cnn_hidden_size`)
            _, h_skip = self.gru_skip(enc_data_skipped) # (1 x -1*`skip` x `skip_hidden_size`)

            h_skip = h_skip.view(batch_size, -1, self.skip * self.skip_hidden_size) # (-1 x 1 x `skip` * `skip_hidden_size`)
            h_skip = self.dropout(h_skip).squeeze(dim=1) # (-1 x `skip`*`skip_hidden_size`)
            output = torch.cat([output, h_skip], dim=1) # (-1 x m+`skip`*`skip_hidden_size`)
        
        output = self.linear_output(output).squeeze(dim=1) # (-1)
        # please please please please please please please please please please be better

        return output


class Model4(nn.Module):
    def __init__(self, T, n, m, skip_hidden_size, skip=10):
        super(Model4, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state

        self.m = m; self.T = T

        # Part of CNN
        self.skip = skip
        self.skip_hidden_size = skip_hidden_size
        self.pt = int(self.T / self.skip)

        # Part of DARNN
        self.encoder = Encoder(T=T, n=n, m=m) # (-1 x T x m)
        
        # Part of LSTNet
        self.cnn_hidden_size = m
        if self.skip > 0:
            self.gru_skip = nn.GRU(self.cnn_hidden_size, self.skip_hidden_size)

        # Part of Result
        self.layer_linear_beta1 = nn.Sequential(
            nn.Linear(self.T, 1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) # (-1 x m x T) -> (-1 x m x 1)

        self.linear_output = nn.Linear(m + self.skip*self.skip_hidden_size, 1) # (-1 x 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):
        # enc: (-1 x T x n)
        batch_size = enc_data.shape[0]
        self.device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.m, device = self.device)
        s0 = torch.zeros(1, batch_size, self.m, device = self.device)

        # 1. Encoder
        H, self.attention_scores_ = self.encoder(enc_data, h_0 = h0, s_0 = s0) # (-1 x T x m)

        # Part of Result
        output = H.permute(0, 2, 1) # (-1 x m x T)
        output = self.layer_linear_beta1(output) # (-1 x m x 1)
        output = output.squeeze(dim=2) # (-1 x m)

        # 2. LSTNet: skip-RNN
        if self.skip > 0:
            enc_data_skipped = H[:, -self.pt*self.skip:, :].permute(0, 2, 1).contiguous() # (-1 x `pt`*`skip` x `m`)
            enc_data_skipped = enc_data_skipped.view(batch_size, self.cnn_hidden_size, self.pt, self.skip) # (-1 x `m` x `pt` x `skip`)
            enc_data_skipped = enc_data_skipped.permute(2, 0, 3, 1).contiguous() # (`pt` x -1 x `skip` x `m`)
            enc_data_skipped = enc_data_skipped.view(self.pt, batch_size * self.skip, self.cnn_hidden_size) # (`pt` x -1*skip x `m`)
            _, h_skip = self.gru_skip(enc_data_skipped) # (1 x -1*`skip` x `skip_hidden_size`)

            h_skip = h_skip.view(batch_size, -1, self.skip * self.skip_hidden_size) # (-1 x 1 x `skip` * `skip_hidden_size`)
            h_skip = self.dropout(h_skip).squeeze(dim=1) # (-1 x `skip`*`skip_hidden_size`)
            output = torch.cat([output, h_skip], dim=1) # (-1 x m+`skip`*`skip_hidden_size`)
        
        # Part of Result
        output = self.linear_output(output).squeeze(dim=1) # (-1)
        # please please please please please please please please please please be better

        return output


#################### Decoder is Used ####################


class DecoderLSTM(nn.Module):
    def __init__(self, m, p):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = m, hidden_size = p, batch_first = True)
        self.initial_state = None
    
    def forward(self, c_t): # c_t: (-1 x 1 x m)
        _, (d_t, s_t) = self.lstm(c_t, self.initial_state) # (1 x -1 x p)
        self.initial_state = (d_t, s_t) # (1 x -1 x p)
        return d_t, s_t # (1 x -1 x p)
    
    def reset_state(self, d_0, s_0):
        self.initial_state = (d_0, s_0) # (1 x -1 x p)


class TemporalAttention(nn.Module):
    def __init__(self, m, p):
        super(TemporalAttention, self).__init__()
        self.W_d = nn.Linear(2*p, m, bias = False)
        self.U_d = nn.Linear(m, m, bias = False)
        self.v_d = nn.Sequential(
            nn.Tanh(), # (-1 x T x m)
            nn.Linear(m, 1, bias=False), # (-1 x T x 1)
            nn.Softmax(dim=1) # (-1 x T x 1)
        )


    def forward(self, d_t, s_t, H): # actually d_(t-1), s_(t-1), ( h(1), ..., h(T) )
        T = H.shape[1]
        d_t = d_t.permute(1, 0, 2) # (-1 x 1 x p)
        s_t = s_t.permute(1, 0, 2) # (-1 x 1 x p)
        query = torch.concat([d_t, s_t], dim = 2) # (-1 x 1 x 2p)
        query = query.repeat(1, T, 1) # (-1 x T x 2p)

        beta_t = self.W_d(query) + self.U_d(H) # (-1 x T x m)
        beta_t = self.v_d(beta_t) # (-1 x T x 1)
        beta_t = beta_t.permute(0, 2, 1) # (-1 x 1 x T)

        return beta_t


class Decoder(nn.Module):
    def __init__(self, T, m, p):
        super(Decoder, self).__init__()
        self.T = T # T'
        self.temporal_attention_score = TemporalAttention(m, p)
        self.lstm = DecoderLSTM(m, p)
    
    def forward(self, H, d_0, s_0):
        d_t, s_t = d_0, s_0
        self.lstm.reset_state(d_0, s_0)

        attention_scores = []
        for t in range(1, self.T+1):
            # finding context vector
            beta_t = self.temporal_attention_score(d_t, s_t, H) # (-1 x 1 x T)
            attention_scores.append(beta_t)
            c_t = beta_t.matmul(H) # (-1 x 1 x m)

            # finding hidden state for next time step
            d_t, s_t = self.lstm(c_t) # (1 x -1 x p)

        attention_scores = torch.cat(attention_scores, dim=1) # (-1 x T x T) | distinguished by rows
        d_t = d_t.permute(1, 0, 2) # (-1 x 1 x p)

        return torch.cat([d_t, c_t], dim=2), attention_scores # (-1 x 1 x m+p)


class Model5(nn.Module):
    def __init__(self, T, n, p, cnn_kernel_height, cnn_hidden_size, skip_hidden_size, skip=10):
        super(Model5, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state
        # p: dimension of Deocder hidden state
        self.p = p; self.T = T
        
        # Part of LSTNet
        self.cnn_kernel_height = cnn_kernel_height
        self.cnn_kernel_width = n
        self.T_modified = T - self.cnn_kernel_height + 1
        self.skip = skip
        self.skip_hidden_size = skip_hidden_size
        self.pt = int(self.T_modified / self.skip)
        self.cnn_hidden_size = cnn_hidden_size
        self.layer_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.cnn_hidden_size,
                kernel_size=(self.cnn_kernel_height, n), stride=1, padding='valid'
            ), # (-1 x 1 x T x n) -> (-1 x `cnn_hidden_size` x T' x 1) | T' = T - `cnn_kernel_height` + 1
            nn.ReLU()
            # nn.Dropout(p=p)
        )
        if self.skip > 0:
            self.gru_skip = nn.GRU(self.cnn_hidden_size, self.skip_hidden_size)

        # Decoder
        self.decoder = Decoder(T = self.T_modified, m = self.cnn_hidden_size, p = p)

        # Part of Result
        self.linear_output = nn.Sequential(
            nn.Linear(self.cnn_hidden_size + p + self.skip*self.skip_hidden_size, p),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(p, 1)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):
        # enc: (-1 x T x n)
        # dec: (-1 x (T-1) x 1)
        batch_size = enc_data.shape[0]
        self.device = next(self.parameters()).device
        d_0 = torch.zeros(1, batch_size, self.p, device=self.device) # (1 x -1 x p)
        s_0 = torch.zeros(1, batch_size, self.p, device=self.device) # (1 x -1 x p)

        # CNN -> going to skip-RNN
        enc_data = enc_data.view(-1, 1, *enc_data.shape[1:]) # (-1 x 1 x T x n)
        enc_data = self.layer_cnn(enc_data) # (-1 x `cnn_hidden_size` x T' x 1)
        enc_data = enc_data.squeeze(dim=3).permute(0, 2, 1) # (-1 x T' x `cnn_hidden_size`)

        # Temporal Attention
        output, self.attention_scores_ = self.decoder(enc_data, d_0=d_0, s_0=s_0) # (-1 x 1 x `cnn_hidden_size`+p)

        # skip-RNN
        if self.skip > 0:
            enc_data_skipped = enc_data[:, -self.pt*self.skip:, :].permute(0, 2, 1).contiguous() # (-1 x `cnn_hidden_size` x `pt`*`skip`)
            enc_data_skipped = enc_data_skipped.view(batch_size, self.cnn_hidden_size, self.pt, self.skip) # (-1 x `cnn_hidden_size` x `pt` x `skip`)
            enc_data_skipped = enc_data_skipped.permute(2, 0, 3, 1).contiguous() # (`pt` x -1 x `skip` x `cnn_hidden_size`)
            enc_data_skipped = enc_data_skipped.view(self.pt, batch_size * self.skip, self.cnn_hidden_size) # (`pt` x -1*skip x `cnn_hidden_size`)
            _, h_skip = self.gru_skip(enc_data_skipped) # (1 x -1*`skip` x `skip_hidden_size`)

            h_skip = h_skip.view(batch_size, -1, self.skip * self.skip_hidden_size) # (-1 x 1 x `skip` * `skip_hidden_size`)
            h_skip = self.dropout(h_skip) # (-1 x 1 x `skip`*`skip_hidden_size`)
            output = torch.cat([output, h_skip], dim=2).squeeze(dim=1) # (-1 x `cnn_hidden_size`+p+`skip`*`skip_hidden_size`)

        output = self.linear_output(output).squeeze(dim=1) # (-1)

        return output


#################### Decoder was Used ####################


class Model6(nn.Module):
    def __init__(self, T, n, m, T_modified):
        super(Model6, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state

        self.m = m; self.T = T

        # Part of Temporal CNN
        self.T_modified = T_modified
        self.layer_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=self.T_modified,
                kernel_size=(T, 1),
                stride=1, padding='valid'
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Part of DARNN
        self.encoder = Encoder(T=self.T_modified, n=n, m=m) # (-1 x T x m)

        self.linear_output = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, 1)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):
        # enc: (-1 x T x n)
        batch_size = enc_data.shape[0]
        self.device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.m, device = self.device)
        s0 = torch.zeros(1, batch_size, self.m, device = self.device)

        # Temporal CNN
        enc_data = enc_data.view(-1, 1, *enc_data.shape[1:]) # (-1 x 1 x T x n)
        enc_data = self.layer_cnn(enc_data) # (-1 x T' x 1 x n)
        enc_data = enc_data.squeeze(dim=2) # (-1 x T' x n)

        # Input Attention
        H, self.attention_scores_ = self.encoder(enc_data, h_0 = h0, s_0 = s0) # (-1 x T x m)
        output = H[:, -1, :] # (-1 x m)
        output = self.linear_output(output).squeeze(dim=1) # (-1)

        return output
    

class Model6_2(nn.Module):
    def __init__(self, T, n, m, T_modified):
        super(Model6_2, self).__init__()
        # T: Time step
        # m: dimension of Encoder hidden state

        self.m = m; self.T = T

        # Part of Temporal CNN
        self.T_modified = T_modified
        self.layer_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=self.T_modified,
                kernel_size=(T, 1),
                stride=1, padding='valid'
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Part of DARNN
        self.encoder = Encoder(T=T, n=n, m=m) # (-1 x T x m)

        self.linear_output1 = nn.Sequential(
            nn.Linear(self.T_modified, int(1.5*self.T_modified)),
            nn.ReLU(),
            nn.Linear(int(1.5*self.T_modified), 1),
            nn.ReLU(),
        )

        self.linear_output2 = nn.Sequential(
            nn.Linear(m, int(1.5*m)),
            nn.ReLU(),
            nn.Linear(int(1.5*m), 1)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, enc_data):
        # enc: (-1 x T x n)
        batch_size = enc_data.shape[0]
        self.device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.m, device = self.device)
        s0 = torch.zeros(1, batch_size, self.m, device = self.device)

        # Input Attention
        H, self.attention_scores = self.encoder(enc_data, h_0=h0, s_0=s0)  # (-1 x T x m)

        # Temporal CNN
        H = H.unsqueeze(dim=1)  # (-1 x 1 x T x m)
        output = self.layer_cnn(H).squeeze(dim=2)  # (-1 x T' x m)

        # Result
        output = self.linear_output1(output.permute(0, 2, 1))  # (-1 x m x 1)
        output = self.linear_output2(output.squeeze(dim=2))  # (-1 x 1)
        output = output.squeeze(dim=1)

        return output