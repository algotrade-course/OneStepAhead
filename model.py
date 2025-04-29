import torch

from transformers import TimeSeriesTransformerForPrediction, AutoformerForPrediction

from typing import (
    Optional,
    List,
    Tuple
)


def save_checkpoint(path, model, optimizer, epoch, loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer,
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, path)    
    print(f"Saved checkpoint with epoch {epoch} and loss {loss:.4f}")


def load_checkpoint(path, model):
    device = next(model.parameters()).device
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = checkpoint["optimizer"]
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded model with epoch {epoch} and loss {loss:.4f}")

    return [model, optimizer, epoch, loss]


class CustomTransformerModel(TimeSeriesTransformerForPrediction):
    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Predict the means and standard deviations of the probability distribution of High and Low values.

        Returns:
            list[torch.Tensor]: means and stds, shape of each: (batch size, prediction_length, input features).
        """
        
        outputs = self(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=True,
        )

        decoder = self.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)

        future_samples = []
        means = []
        stds = []
        dfs = []

        # greedy decoding
        for k in range(self.config.prediction_length):
            lagged_sequence = self.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1)

            dec_output = decoder(inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden)
            dec_last_hidden = dec_output.last_hidden_state

            params = self.parameter_projection(dec_last_hidden[:, -1:])
            distr = self.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
            next_sample = distr.sample()

            means.append(distr.mean[::num_parallel_samples])
            stds.append(distr.stddev[::num_parallel_samples])
            dfs.append(distr.base_dist.base_dist.df[::num_parallel_samples])

            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        # concat_future_samples = torch.cat(future_samples, dim=1)

        # return SampleTSPredictionOutput(
        #     sequences=concat_future_samples.reshape(
        #         (-1, num_parallel_samples, self.config.prediction_length) + self.target_shape,
        #     )
        # )

        concat_means = torch.cat(means, dim=1).reshape(
            (-1, self.config.prediction_length) + self.target_shape,
        )
        concat_stds = torch.cat(stds, dim=1).reshape(
            (-1, self.config.prediction_length) + self.target_shape,
        )
        concat_dfs = torch.cat(dfs, dim=1).reshape(
            (-1, self.config.prediction_length) + self.target_shape,
        )
        return concat_means, concat_stds, concat_dfs


class CustomAutoformerModel(AutoformerForPrediction):
    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        r"""
        Predict the means and standard deviations of the probability distribution of High and Low values.

        Returns:
            list[torch.Tensor]: means and stds, shape of each: (batch size, prediction_length, input features).
        """
        outputs = self(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=None,
            future_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )

        decoder = self.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        time_features = torch.cat((past_time_features, future_time_features), dim=1)

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)

        lagged_sequence = self.model.get_lagged_subsequences(
            sequence=repeated_past_values, subsequences_length=self.config.context_length
        )
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
        seasonal_input, trend_input = self.model.decomposition_layer(reshaped_lagged_sequence)

        mean = torch.mean(reshaped_lagged_sequence, dim=1).unsqueeze(1).repeat(1, self.config.prediction_length, 1)
        zeros = torch.zeros(
            [reshaped_lagged_sequence.shape[0], self.config.prediction_length, reshaped_lagged_sequence.shape[2]],
            device=reshaped_lagged_sequence.device,
        )

        decoder_input = torch.cat(
            (
                torch.cat((seasonal_input[:, -self.config.label_length :, ...], zeros), dim=1),
                repeated_features[:, -self.config.prediction_length - self.config.label_length :, ...],
            ),
            dim=-1,
        )
        trend_init = torch.cat(
            (
                torch.cat((trend_input[:, -self.config.label_length :, ...], mean), dim=1),
                repeated_features[:, -self.config.prediction_length - self.config.label_length :, ...],
            ),
            dim=-1,
        )
        decoder_outputs = decoder(
            trend=trend_init, inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden
        )
        decoder_last_hidden = decoder_outputs.last_hidden_state
        trend = decoder_outputs.trend
        params = self.output_params(decoder_last_hidden + trend)
        distr = self.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
        future_samples = distr.sample()

        means = distr.mean[::num_parallel_samples]
        stds = distr.stddev[::num_parallel_samples]

        # print(means.size())
        # print(stds.size())

    