import torch
import torch.nn as nn

from modules.models import *
from .image_phasor import ImagePhasor


def get_model(model_args):
    model_type = model_args["model_type"]
    if model_type == "implicit":
        implicit_layer = DEQImplicitLayer(**model_args,)

        model = DEQImplicitNN(
            implicit_layer=implicit_layer, output_channels=model_args["out_channels"],
        )
    elif model_type == "siren":
        model = SIREN(
            in_channels=model_args["in_channels"],
            interm_channels=model_args["interm_channels"],
            out_channels=model_args["out_channels"],
            scale=model_args["input_scale"],
            n_layers=model_args["n_layers"],
        )
    elif model_type == "ffn":
        model = FFN(
            in_channels=model_args["in_channels"],
            interm_channels=model_args["interm_channels"],
            out_channels=model_args["out_channels"],
            n_layers=model_args["n_layers"],
        )

    elif model_type == "phasor":
        model = ImagePhasor(
            H=model_args["H"],
            W=model_args["W"],
            ch=model_args["ch"],
            using_mlp=True,
            feat_mode=model_args["feat_mode"],
            feat_cfg=model_args["feat_cfg"],
            mlp_cfg=model_args["mlp_cfg"],
        )

    return model


def construct_model_args(
    model_type,
    n_layers,
    in_channels,
    interm_channels,
    out_channels,
    input_scale=256.0,
    use_implicit=False,
    filter_type="fourier",
    filter_options={},
    norm_type="none",
    tol=1e-3,
    forward_solver="forward_iter",
    backward_solver="forward_iter",
):
    _dict = {
        "model_type": model_type,
        "n_layers": n_layers,
        "in_channels": in_channels,
        "interm_channels": interm_channels,
        "out_channels": out_channels,
        "input_scale": input_scale,
        "one_pass": not use_implicit,
        "tol": tol,
        "forward_solver": forward_solver,
        "backward_solver": backward_solver,
        "filter_type": filter_type,
        "filter_options": filter_options,
        "norm_type": norm_type,
        "init": "default" if not use_implicit else "deq",
    }
    return _dict


def get_summary_dict(
    in_channels,
    out_channels=None,
    filter_options=["gabor", "fourier", "siren_like"],
    size_options=[(1, 256), (1, 512), (4, 256)],
    input_scale=256.0,
):
    if out_channels is None:
        out_channels = in_channels

    deq_options = [True, False]
    _dict = {}

    for size in size_options:
        for filter_t in filter_options:
            for deq in deq_options:
                n_layer, interm_channels = size

                deq_tag = "DEQ-" if deq else ""
                filter_tag = filter_t.capitalize() + "-"

                interm_channel_tag = "{:d}D".format(interm_channels)
                tag = (
                    deq_tag
                    + filter_tag
                    + "MFN"
                    + str(n_layer)
                    + "L"
                    + interm_channel_tag
                )

                _dict[tag] = {
                    "config": construct_model_args(
                        "implicit",
                        n_layer,
                        in_channels,
                        interm_channels,
                        out_channels,
                        # input_scale=input_scale / 2 if filter_t == 'siren_like' else input_scale,
                        input_scale=input_scale,
                        use_implicit=deq,
                        filter_type=filter_t,
                        filter_options={"alpha": 3.0},
                        norm_type="spectral_norm",
                        forward_solver="forward_iter",
                        backward_solver="onestep",
                    ),
                    "results": {},
                    "state_dicts": [],
                    "misc": {},
                    "finished": False,
                }
    return _dict


def get_summary_dict_phasor(H, W, ch, cfgs_list):
    _dict = {}

    for cfg in cfgs_list:
        mode = cfg.Image.mode
        tag = mode

        _dict[tag] = {
            "config": {
                "model_type": "phasor",
                "H": H,
                "W": W,
                "ch": ch,
                "feat_mode": mode,
                "feat_cfg": cfg.Encoding,
                "mlp_cfg": cfg.MLP,
            },
            "results": {},
            "state_dicts": [],
            "misc": {},
            "finished": False,
        }
    return _dict
