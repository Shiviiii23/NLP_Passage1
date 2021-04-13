# This list of imports is likely incomplete --- add anything you need.
# TODO: Your code here.
import torch.nn as nn
import torch

class AttentionRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size,
                 dropout):
        """
        Parameters
        ----------
        embedding_matrix: FloatTensor
            FloatTensor matrix of shape (num_words, embedding_dim),
            where each row of the matrix is a word vector for the
            associated word index.

        hidden_size: int
            The size of the hidden state in the RNN.

        dropout: float
            The dropout rate.
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        self.embedding_matrix = embedding_matrix
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)
        super(AttentionRNN, self).__init__()

        # Create Embedding object
        # TODO: Your code here.
        self.embedding = nn.Embedding(self.num_embedding_words,
                                      self.embedding_dim, padding_idx=0)
        # Load our embedding matrix weights into the Embedding object,
        # and make them untrainable (requires_grad=False)
        # TODO: Your code here.
        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)
        # Make a RNN to encode the passage. Note that batch_first=True.
        # TODO: Your code here.
        self.passage_gru = nn.GRU(self.embedding_dim, self.hidden_size//2, batch_first = True, bidirectional = True, dropout = dropout)
        # Make a RNN to encode the question. Note that batch_first=True.
        # TODO: Your code here.
        self.question_gru = nn.GRU(self.embedding_dim, self.hidden_size//2, batch_first = True, bidirectional = True, dropout = dropout)
        # Affine transform for attention.
        # TODO: Your code here.
        self.attention = nn.Linear(3 * self.hidden_size, 1)
        # Affine transform for predicting start index.
        # TODO: Your code here.
        self.start_output_projection = nn.Linear(3 * self.hidden_size, 1)
        # Affine transform for predicting end index.
        # TODO: Your code here.
        self.end_output_projection = nn.Linear(3 * self.hidden_size, 1)
        # Dropout layer
        # TODO: Your code here.

        # Stores the number of gradient updates performed.
        self.global_step = 0

    def forward(self, passage, question):
        """
        The forward pass of the RNN-based model with attention.

        Parameters
        ----------
        passage: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, passage_length)
            representing the words in the passage for each batch.

        question: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, question_length)
            representing the words in the question for each batch.

        Returns
        -------
        An output dictionary consisting of:
        start_logits: Variable(FloatTensor)
            The first element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        end_logits: Variable(FloatTensor)
            The second element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        softmax_start_logits: Variable(FloatTensor)
            The third element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer begins.
            Masked indices have probability mass of -inf.

        softmax_end_logits: Variable(FloatTensor)
            The fourth element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer end.
            Masked indices have probability mass of -inf.
        """
        # Mask: FloatTensor with 0 in positions that are
        # padding (word index 0) and 1 in positions with actual words.
        # Make a mask for the passage. Shape: ?
        # TODO: Your code here.
        passage_mask = (passage != 0).type(
            torch.cuda.FloatTensor if passage.is_cuda else
            torch.FloatTensor)
        # Make a mask for the question. Shape: ?
        # TODO: Your code here.
        question_mask = (question != 0).type(
            torch.cuda.FloatTensor if question.is_cuda else
            torch.FloatTensor)
        # Make a LongTensor with the length (number non-padding words
        # in) each passage.
        # Shape: ?
        # TODO: Your code here.
        passage_length = passage_mask.sum(dim=1)
        # Make a LongTensor with the length (number non-padding words
        # in) each question.
        # Shape: ?
        # TODO: Your code here.
        question_length = question_mask.sum(dim=1)
        # Part 1: Embed the passages and the questions.
        # 1.1 Embed the passage.
        # TODO: Your code here.
        # Shape: ?
        embedded_passage = self.embedding(passage)
        # 1.2. Embed the question.
        # TODO: Your code here.
        # Shape: ?
        embedded_question = self.embedding(question)
        # Part 2. Encode the embedded passages with the RNN.
        # 2.1. Sort embedded passages by decreasing order of passage_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        # TODO: Your code here.
        sort_passage, sort_passage_length, old_passage_length, _ = sort_batch_by_length(embedded_passage, passage_length)
        # 2.2. Pack the passages with torch.nn.utils.rnn.pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.
        packed_passage = torch.nn.utils.rnn.pack_padded_sequence(sort_passage, sort_passage_length, batch_first=True)
        # 2.3. Encode the packed passages with the RNN.
        # TODO: Your code here.
        output, hn = self.passage_gru(packed_passage)
        # 2.4. Unpack (pad) the passages with
        # torch.nn.utils.rnn.pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.
        passage_unpacked, len_unpacked = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # 2.5. Unsort the unpacked, encoded passage to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.
        restoredp = passage_unpacked.index_select(0, old_passage_length)
        # Part 3. Encode the embedded questions with the RNN.
        # 3.1. Sort the embedded questions by decreasing order
        #      of question_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        # TODO: Your code here.
        sort_question, sort_question_length, old_question_length, _ = sort_batch_by_length(embedded_question, question_length)
        # 3.2. Pack the questions with pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.
        packed_question = torch.nn.utils.rnn.pack_padded_sequence(sort_question, sort_question_length, batch_first=True)
        # 3.3. Encode the questions with the RNN.
        # TODO: Your code here.
        outputq, hnq = self.question_gru(packed_question)
        # 3.4. Unpack (pad) the questions with pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.
        question_unpacked, len_unpackedq = torch.nn.utils.rnn.pad_packed_sequence(outputq, batch_first=True)

        # 3.5. Unsort the unpacked, encoded question to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.
        restoredq = question_unpacked.index_select(0, old_question_length)
        # Part 4. Calculate attention weights and attend to question.
        # 4.1. Expand the encoded question to shape suitable for attention.
        # Hint: Think carefully about what the shape of the attention
        # input vector should be. torch.unsqueeze and torch.expand
        # might be useful.
        # Shape: ?
        # TODO: Your code here.
        #get the max passage size by getting size of passage mask (2nd value)
        maxp_size = passage_mask.size()
        tiled_encoded_q = question_unpacked.unsqueeze(dim=1).expand_as(
            -1, maxp_size, -1, -1)
        # 4.2. Expand the encoded passage to shape suitable for attention.
        # Hint: Think carefully about what the shape of the attention
        # input vector should be. torch.unsqueeze and torch.expand
        # might be useful.
        # Shape: ?
        # TODO: Your code here.
        maxq_size = question_mask.size()
        tiled_encoded_p = passage_unpacked.unsqueeze(dim=1).expand_as(-1, -1, maxq_size, -1)

        # 4.3. Build attention_input. This is the tensor passed through
        # the affine transform.
        # Hint: Think carefully what the shape of this tensor should be.
        # torch.cat might be useful.
        # Shape: ?
        # TODO: Your code here.

        combined_x_q = torch.cat([restoredp, tiled_encoded_q,
                                  restoredp * tiled_encoded_q], dim=-1)

        # 4.4. Apply affine transform to attention input to get
        # attention logits. You will need to slightly reshape it
        # into a tensor of the shape you expect.
        # Shape: ?
        # TODO: Your code here.

        attention_logits = self.attention(combined_x_q).squeeze(-1)

        # 4.5. Masked-softmax the attention logits over the last dimension
        # to normalize and make the attention logits a proper
        # probability distribution.
        # Hint: allennlp.nn.util.last_dim_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.
        masked_softmax = allennlp.nn.util.masked_softmax(attention_logits, question_mask)

        # 4.6. Use the attention weights to get a weighted average
        # of the RNN output from encoding the question for each
        # passage word.
        # Hint: torch.bmm might be helpful.
        # Shape: ?
        # TODO: Your code here.
        weighted_average = torch.bmm(masked_softmax, restoredq)

        # Part 5: Combine the passage and question representations by
        # concatenating the passage and question representations with
        # their product.
        # 5.1. Concatenate to make the combined representation.
        # Hint: Use torch.cat
        # Shape: ?
        # TODO: Your code here.

        combined_rep = torch.cat([restoredp, weighted_average,
                                restoredp*weighted_average], dim=-1)
        # Part 6: Compute logits for answer start index.

        # 6.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.
        start_logits = self.start_output_projection(combined_x_q).squeeze(-1)

        # 6.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your start_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
        # Shape: ?
        # TODO: Your code here.
        start_logits = replace_masked_values(start_logits, passage_mask, -1e7)

        # 6.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_start_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.
        softmax_start_logits = masked_log_softmax(start_logits, passage_mask)

        # Part 7: Compute logits for answer end index.

        # 7.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.
        end_logits = self.end_output_projection(combined_x_q).squeeze(-1)

        # 7.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your end_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
        # Shape: ?
        # TODO: Your code here.
        end_logits = replace_masked_values(end_logits, passage_mask, -1e7)

        # 7.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_end_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.
        softmax_end_logits = masked_log_softmax(end_logits, passage_mask)

        # Part 8: Output a dictionary with the start_logits, end_logits,
        # softmax_start_logits, softmax_end_logits.
        # TODO: Your code here. Remove the NotImplementedError below.
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "softmax_start_logits": softmax_start_logits,
            "softmax_end_logits": softmax_end_logits,
        }
        raise NotImplementedError
