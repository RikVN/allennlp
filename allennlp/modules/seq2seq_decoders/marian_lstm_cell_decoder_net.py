from typing import Tuple, Dict, Optional
from overrides import overrides

import torch
from torch.nn import LSTMCell

from allennlp.modules import Attention
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet
from allennlp.nn import util
from allennlp.common.util import fixed_seeds

@DecoderNet.register("marian_lstm_cell")
class MarianLstmCellDecoderNet(DecoderNet):
    """
    This decoder net implements simple decoding network with LSTMCell and Attention.

    Parameters
    ----------
    decoding_dim : ``int``, required
        Defines dimensionality of output vectors.
    target_embedding_dim : ``int``, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """

    def __init__(self,
                 decoding_dim: int,
                 target_embedding_dim: int,
                 attention: Optional[Attention] = None,
                 bidirectional_input: bool = False,
                 extra_lstm=False) -> None:

        super().__init__(decoding_dim=decoding_dim,
                         target_embedding_dim=target_embedding_dim,
                         decodes_parallel=False)
        # Set seeds again, better reproducibility
        fixed_seeds()
        # In this particular type of decoder output of previous step passes directly to the input of current step
        # We also assume that decoder output dimensionality is equal to the encoder output dimensionality
        decoder_input_dim = self.target_embedding_dim

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention
        self.extra_lstm = extra_lstm

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # First LSTM cell maps from the decoder state (2x input dim because of the features) to the decoding dimension
        self._decoder_cell1 = LSTMCell(self.decoding_dim, self.decoding_dim)
        # Second LSTM cell maps from 2 (bidirectional) * 2 (features) * input_dim to the decoding dimension
        self._decoder_cell2 = LSTMCell(int(2 * self.decoding_dim) + self.target_embedding_dim, self.decoding_dim)
        self._bidirectional_input = bidirectional_input

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.Tensor = None,
                                encoder_outputs: torch.Tensor = None,
                                encoder_outputs_mask: torch.Tensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state,
                                        encoder_outputs,
                                        encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    def init_decoder_state(self, encoder_out: Dict[str, torch.LongTensor], extra_enc_layer=False) -> Dict[str, torch.Tensor]:

        batch_size, _ = encoder_out["source_mask"].size()

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        # Concatenate all encoder states we created before
        init_states = []
        return_dic = {}
        # Always do token output first
        for key in encoder_out:
            if key == "encoder_outputs":
                mask_str = "source_mask"
                cur_state = util.get_averaged_encoder_states(encoder_out[key], encoder_out["source_mask"], bidirectional=self._bidirectional_input)
                init_states.append(cur_state)
                return_dic["decoder_init"] = cur_state
        # Then check for features (max 1 currently)
        for key in encoder_out:
            if key.startswith("encoder_outputs") and key != "encoder_outputs":
                ident = key.replace('encoder_outputs','')
                mask_str = "source_mask" + ident
                cur_state = util.get_averaged_encoder_states(encoder_out[key], encoder_out[mask_str], bidirectional=self._bidirectional_input)
                init_states.append(cur_state)
                return_dic["decoder_init" + ident] = cur_state

        # Concatenate state (max 2 for now)
        final_encoder_output = torch.cat(init_states, dim=-1)
        # Extra layer between encoder and decoder -- map to decoder space (original length)
        # So always has to be there for two encoders
        final_encoder_output = extra_enc_layer(final_encoder_output)
        return_dic["decoder_hidden"] = final_encoder_output # shape: (batch_size, decoder_output_dim)
        return_dic["decoder_context"] = final_encoder_output.new_zeros(batch_size, int(self.decoding_dim)) # shape: (batch_size, decoder_output_dim)
        return return_dic

    @overrides
    def forward(self,
                previous_state: Dict[str, torch.Tensor],
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                previous_steps_predictions: torch.Tensor,
                previous_steps_mask: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor],
                                                                             torch.Tensor]:
        return_dic = {}
        decoder_hidden = previous_state['decoder_hidden']
        decoder_context = previous_state['decoder_context']
        # First put inputs through LSTM to create temporary states if we want
        if self.extra_lstm:
            decoder_hidden, decoder_context = self._decoder_cell1(decoder_hidden, (decoder_hidden, decoder_context))
        # shape: (group_size, output_dim)
        last_predictions_embedding = previous_steps_predictions[:, -1]

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            # Loop over the keys of the state to find how many features we are doing
            # Bit hacky but it should work
            att_inputs = []
            for key in previous_state:
                if key.startswith("encoder_outputs"):
                    # Filter embeddings so it only contains the information relevant for them
                    ident = key.replace('encoder_outputs','')
                    if not ident:
                        mask_str = "source_mask"
                    else:
                        mask_str = "source_mask" + ident
                    # Apply attention
                    cur_att = self._prepare_attended_input(decoder_hidden, previous_state[key], previous_state[mask_str])
                    att_inputs.append(cur_att)
            # Concatenate the attention vectors (max 2), and put those through an LSTM again
            attended_input = torch.cat(att_inputs, dim=-1)
            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, last_predictions_embedding), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = last_predictions_embedding

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell2(decoder_input, (decoder_hidden, decoder_context))
        return_dic["decoder_hidden"] = decoder_hidden
        return_dic["decoder_context"] = decoder_context
        return return_dic, decoder_hidden
