import torch
import torch.nn as nn

class TESLSTM(nn.Module):
    ''' LSTM cell model to predict the next character in a name.

    The LSTM model takes as input the race tensor, gender tensor, single
    character tensor, the previous hidden and cell states and outputs the
    next character tensor.
    '''

    def __init__(self, input_size, hidden_size, output_size):
        ''' Initializes the parameters of the LSTM cell model

        Parameters
        ----------
        input_size: int
            Number of input units of the LSTM.
        hidden_size: int
            Number of hidden units and cell memory units of the LSTM.
        output_size: int
            Number of output units of the LSTM.
        '''
        super(TESLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.05)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, t_race, t_gender, t_char, t_hidden, t_cell):
        ''' Forward pass execution for the LSTM model.

        Input tensors (`t_race`, `t_gender`, `t_char`) should be 3rd rank.
        Hidden tensors (`t_hidden` and `t_cell`) should be 2nd rank.

        The forward pass concatenates the input tensors and performs
        a pass through the LSTM cell and through a fully-connected layer
        before a softmax layer at the output units.

        Parameters
        ----------
        t_race: torch.Tensor
            The tensor of batches of race vectors.
        t_gender: torch.Tensor
            The tensor of batches of gender vectors.
        t_char: torch.Tensor
            The tensor of batches of single character vectors of a name.
        t_hidden: torch.Tensor
            The tensor of batches of hidden units.
        t_cell: torch.Tensor
            The tensor of batches of cell memory units.
        '''
        t_input = torch.cat((t_race, t_gender, t_char), dim=2)
        t_input = t_input.view(-1, self.input_size)
        t_hidden, t_cell = self.lstm(t_input, (t_hidden, t_cell))

        t_output = self.fc(self.dropout(t_hidden))
        t_output = self.softmax(t_output)

        return t_output, t_hidden, t_cell

    def init_hidden(self, batch_size):
        ''' Initializes the LSTM's hidden and memory cell values.

        The LSTM starts with a random hidden state and a zero memory cell
        state as the neural network needs to be able to generate new names.
        With the random initial state, the network needs to learn to
        generalize this process when training.

        Parameters
        ----------
        batch_size: int
            Size of the current batch of tensors.
        '''
        t_hidden = torch.randn(batch_size, self.hidden_size)
        t_cell = torch.zeros(batch_size, self.hidden_size)

        return t_hidden, t_cell


if __name__ == '__main__':
    model = TESLSTM(10, 20, 5)

    t_race = torch.randn(32, 1, 3)
    t_gender = torch.randn(32, 1, 2)
    t_name = torch.randn(32, 30, 5)
    t_hidden, t_cell = model.init_hidden(32)

    print(model(t_race, t_gender, t_name[:, 0:1], t_hidden, t_cell))
