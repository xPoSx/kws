import torch
from src.baseline_model import CRNN


class StreamingCRNN(CRNN):
    def __init__(self, max_window_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_window_length = max_window_length

    def forward(self, input):
        if self.training:
            return super(StreamingCRNN, self).forward(input)
        else:
            gru_buf = []
            hidden_state = None
            outputs = []
            i = 0
            while i < input.shape[1] - self.config.kernel_size[1]:

                conv_input = input[:, i:i + self.config.kernel_size[1]].unsqueeze(0)

                conv_input = conv_input.unsqueeze(dim=1)
                conv_output = self.conv(conv_input).transpose(-1, -2)
                gru_output, hidden_state = self.gru(conv_output, hidden_state)
                gru_buf.append(gru_output)

                if i + self.config.kernel_size[1] >= self.max_window_length:
                    attn_inp = torch.cat(gru_buf, dim=1)
                    contex_vector = self.attention(attn_inp)
                    outputs.append(self.classifier(contex_vector))
                    gru_buf.pop(0)
                i += self.config.stride[1]

            return outputs
