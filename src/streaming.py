import torch
import torch.nn.functional as F


class StreamingCRNN:
    def __init__(self, max_window_length, step_size):
        self.max_window_length = max_window_length
        self.step_size = step_size

    @torch.no_grad()
    def validation(self, model, input):
        model.eval()
        hidden_state = None
        probs = []

        for i in range(0, input.shape[1], self.step_size):
            if i + 20 > input.shape[1]:
                break

            output, hidden_state = model(input[:, i:i + self.max_window_length].unsqueeze(0), hidden_state, True)
            probs.append(F.softmax(output, dim=-1)[0][1])

        return probs
