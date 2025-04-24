import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.Autoformer_EncDec import series_decomp


class TrendMixing(nn.Module):
    def __init__(self, configs):
        super(TrendMixing, self).__init__()
        self.trend_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])
        self.season_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, trend_list, season_list):

        out_high_1 = season_list[0]
        out_low_1 = season_list[1]
        out_season_list = [out_high_1]

        for i in range(len(season_list) - 1):
            out_low_res = self.season_layers[i](out_high_1.permute(0, 2, 1))
            out_low_1 = out_low_1 + out_low_res.permute(0, 2, 1)
            out_high_1 = out_low_1
            if i + 2 <= len(season_list) - 1:
                out_low_1 = season_list[i + 2]
            out_season_list.append(out_high_1)

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.trend_layers[i](out_low.permute(0, 2, 1))
            out_high = out_high + out_high_res.permute(0, 2, 1)
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        return out_trend_list, out_season_list


class Patch(nn.Module):
    def __init__(self, configs):
        super(Patch, self).__init__()
        self.n_history = 1
        self.configs = configs
        self.layers = configs.down_sampling_layers
        self.win = configs.down_sampling_window
        self.dropout = nn.Dropout(configs.dropout)
        self.norm = nn.BatchNorm1d(configs.seq_len * configs.d_model)
        self.norm1 = [nn.BatchNorm1d(self.n_history * self.win ** (i + 1) * configs.d_model) for i in
                      range(self.layers)]
        self.agg = [nn.Linear(self.n_history * self.win ** (i + 1), self.win ** (i + 1)) for i in range(self.layers)]
        self.linear = [nn.Linear(configs.seq_len, configs.seq_len // (self.win ** (i + 1))) for i in range(self.layers)]

    def forward(self, x):
        device = x.device
        self.norm = self.norm.to(device)
        for i in range(len(self.norm1)):
            self.norm1[i] = self.norm1[i].to(device)
            self.agg[i] = self.agg[i].to(device)
            self.linear[i] = self.linear[i].to(device)
        length_list = [
            self.win ** (i + 1) for i in range(self.layers)
        ]
        patch_list = [x]     # 128*21 96 16
        for j, patch_size in zip(range(len(length_list)), length_list):
            ori = torch.transpose(x, 1, 2)
            ori = self.norm(torch.flatten(ori, 1, -1)).reshape(ori.shape)
            out_put = torch.zeros_like(ori)
            out_put[:, :, :self.n_history * patch_size] = ori[:, :, :self.n_history * patch_size]
            for i in range(self.n_history * patch_size, ori.shape[-1], patch_size):
                in_put = out_put[:, :, i - self.n_history * patch_size: i]
                in_put = self.norm1[j](torch.flatten(in_put, 1, -1)).reshape(in_put.shape)
                in_put = F.relu(self.agg[j](in_put))
                in_put = self.dropout(in_put)
                tmp = in_put + ori[:, :, i: i + patch_size]
                out_put[:, :, i: i + patch_size] = tmp
            out_put = self.linear[j](out_put).transpose(1, 2)
            patch_list.append(out_put)
        return patch_list


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.x_mark_dec = None
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
        self.down_sample_window = configs.down_sampling_window
        self.down_sample_layers = configs.down_sampling_layers

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=False)
                for i in range(self.down_sample_layers + 1)
            ]
        )

        self.trend_mixing = TrendMixing(configs)

        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_ff),
            nn.GELU(),
            nn.Linear(in_features=self.d_ff, out_features=self.d_model),
        )

        self.patch_x = Patch(configs)

        self.decompsition = series_decomp(configs.moving_avg)

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

    def mutil_scale_decomposition(self, x_enc, x_mark):
        down_pool = torch.nn.AvgPool1d(self.down_sample_window)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_ori = x_mark

        x_enc_list = []
        x_mark_list = []
        x_enc_list.append(x_enc.permute(0, 2, 1))
        x_mark_list.append(x_mark)

        for i in range(self.down_sample_layers):
            x_enc_sample = down_pool(x_enc_ori)
            x_enc_list.append(x_enc_sample.permute(0, 2, 1))
            x_enc_ori = x_enc_sample

            if x_mark_ori is not None:
                x_mark_list.append(x_mark_ori[:, ::self.down_sample_window, :])
                x_mark_ori = x_mark_ori[:, ::self.down_sample_window, :]

        return x_enc_list, x_mark_list

    def pre_enc(self, x_list):
        return (x_list, None)

    def future_mutil_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        x_list = x_list[0]
        device = enc_out_list[0].device
        adaptive_params = [torch.nn.Parameter(torch.ones(self.configs.c_out, device=device)) for _ in
                           range(len(x_list))]
        for i, (enc_out, adaptive_param) in enumerate(zip(enc_out_list, adaptive_params)):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out = dec_out * torch.sigmoid(adaptive_param).unsqueeze(0)
            dec_out_list.append(dec_out)
        return dec_out_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, N, T = x_enc.size()
        x_mark_dec = x_mark_dec.repeat(N, 1, 1)
        self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.mutil_scale_decomposition(x_enc, x_mark_enc)  # 128 96/48/24/12 21;128 96/48/24/12 4

        x_list = []
        x_mark_list = []
        for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_mark = x_mark.repeat(N, 1, 1)
            x_list.append(x)
            x_mark_list.append(x_mark)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
            enc_out = self.enc_embedding(x, x_mark)
            enc_out_list.append(enc_out)
        length_list = []
        for x in enc_out_list:
            _, length, _ = x.size()
            length_list.append(length)
        season_list = []
        trend_list = []
        for enc_out in enc_out_list:
            season, trend = self.decompsition(enc_out)
            season_list.append(season)
            trend_list.append(trend)
        trend_list, season_list = self.trend_mixing(trend_list, season_list)
        patch_list = self.patch_x(enc_out_list[0])
        for i, patch_x, trend_x, season_x, enc_out, length in zip(range(len(patch_list)), patch_list, trend_list, season_list, enc_out_list, length_list):
            out = enc_out + self.cross_layer(patch_x + trend_x + season_x)
            enc_out_list[i] = out[:, :length, :]    # 128*21 96/48/24/12 16
        dec_out_list = self.future_mutil_mixing(B, enc_out_list, x_list)   # 128 96 21
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)   # 128 96 21
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out_list
