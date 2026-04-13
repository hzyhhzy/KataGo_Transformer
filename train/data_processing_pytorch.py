import logging
import os
import itertools

import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

import modelconfigs

AXIS_PERMUTATIONS_3D = list(itertools.permutations((0, 1, 2)))

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    symmetry_type: str,
    include_meta: bool,
    history_matrices_type: str,
    model_config: modelconfigs.ModelConfig,
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    history_matrices_type = (history_matrices_type or "").strip().lower()
    enable_history_matrices = history_matrices_type == "go"
    is_gomoku_history = history_matrices_type == "gomoku"
    assert history_matrices_type in ("", "none", "go", "gomoku"), f"Unknown history_matrices_type {history_matrices_type}"
    if enable_history_matrices:
        (h_base,h_builder) = build_history_matrices(model_config, device)
    if is_gomoku_history:
        assert num_bin_features == 22, f"gomoku history requires 22 spatial channels, got {num_bin_features}"
        assert num_global_features == 39, f"gomoku history requires 39 global channels, got {num_global_features}"

    def load_npz_file(npz_file):
        with np.load(npz_file) as npz:
            if "binaryInputNCLPacked" in npz:
                binary_input_packed = npz["binaryInputNCLPacked"]
            else:
                # not sure: assuming some datasets may still keep the old packed key name after switching to NCL.
                binary_input_packed = npz["binaryInputNCHWPacked"]
            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
            globalTargetsNC = npz["globalTargetsNC"]
            if include_meta:
                metadataInputNC = np.zeros(
                    (
                        binary_input_packed.shape[0],
                        modelconfigs.get_num_meta_encoder_input_features(model_config),
                    ),
                    dtype=np.float32,
                )
            else:
                metadataInputNC = None
        del npz

        binaryInputNCL = np.unpackbits(binary_input_packed, axis=2)
        assert len(binaryInputNCL.shape) == 3
        expected_l = pos_len * pos_len * pos_len
        assert binaryInputNCL.shape[2] == ((expected_l + 7) // 8) * 8
        binaryInputNCL = binaryInputNCL[:, :, :expected_l].astype(np.float32)

        assert binaryInputNCL.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        return (npz_file, binaryInputNCL, globalInputNC, policyTargetsNCMove, globalTargetsNC, metadataInputNC)

    if not npz_files:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_npz_file, npz_files[0])

        for next_file in (npz_files[1:] + [None]):
            (npz_file, binaryInputNCL, globalInputNC, policyTargetsNCMove, globalTargetsNC, metadataInputNC) = future.result()

            num_samples = binaryInputNCL.shape[0]
            # Just discard stuff that doesn't divide evenly
            num_whole_steps = num_samples // (batch_size * world_size)

            #logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            if next_file is not None:
                #logging.info(f"Preloading {next_file} while processing this file")
                future = executor.submit(load_npz_file, next_file)

            for n in range(num_whole_steps):
                start = (n * world_size + rank) * batch_size
                end = start + batch_size

                batch_binaryInputNCL = torch.from_numpy(binaryInputNCL[start:end]).to(device)
                batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
                batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
                batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)

                if enable_history_matrices:
                    (batch_binaryInputNCL, batch_globalInputNC) = apply_history_matrices(
                        model_config, batch_binaryInputNCL, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                    )
                if is_gomoku_history:
                    zero_mask = (torch.rand((batch_binaryInputNCL.shape[0],), device=batch_binaryInputNCL.device) < 0.3).to(batch_binaryInputNCL.dtype)
                    batch_binaryInputNCL[:, 6, :] *= (1.0 - zero_mask).view(-1, 1)
                    batch_globalInputNC[:, 1] *= (1.0 - zero_mask)


                
                if symmetry_type is not None and symmetry_type!="" and symmetry_type!="none":
                    if symmetry_type == "xyt":
                        allowed_symms = list(range(48))
                    else:
                        assert False, f"Unknown or unsupported 3D data symmetry type {symmetry_type}"
                        
                    symm = allowed_symms[int(rand.integers(0,len(allowed_symms)))]
                    batch_binaryInputNCL = apply_symmetry(batch_binaryInputNCL, symm)
                    batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)

                batch_binaryInputNCL = batch_binaryInputNCL.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()

                batch = dict(
                    binaryInputNCL = batch_binaryInputNCL,
                    globalInputNC = batch_globalInputNC,
                    policyTargetsNCMove = batch_policyTargetsNCMove,
                    globalTargetsNC = batch_globalTargetsNC,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC

                yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:, :, :-1].view((batch_size, channels, pos_len, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len * pos_len * pos_len),
        tensor[:, :, -1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply one of the 48 cube symmetries to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be transformed. (..., H, W, Z)
        symm (int): 0..47 = 6 axis permutations * 8 flip combinations.
    """
    assert tensor.shape[-1] == tensor.shape[-2] == tensor.shape[-3]
    assert 0 <= symm < 48, f"3D symmetry id out of range: {symm}"

    perm_idx = symm // 8
    flip_bits = symm % 8
    spatial_perm = AXIS_PERMUTATIONS_3D[perm_idx]

    permute_order = list(range(tensor.dim() - 3)) + [tensor.dim() - 3 + axis for axis in spatial_perm]
    tensor = tensor.permute(permute_order)

    flip_dims = []
    if flip_bits & 1:
        flip_dims.append(tensor.dim() - 3)
    if flip_bits & 2:
        flip_dims.append(tensor.dim() - 2)
    if flip_bits & 4:
        flip_dims.append(tensor.dim() - 1)
    if flip_dims:
        tensor = tensor.flip(flip_dims)
    return tensor


def build_history_matrices(model_config: modelconfigs.ModelConfig, device):
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    assert num_bin_features == 22, "Currently this code is hardcoded for this many features"

    h_base = torch.diag(
        torch.tensor(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            device=device,
            requires_grad=False,
        )
    )
    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroing them we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prev move, we enable feature 9 and 15
    h0[9, 9] = 1.0  # Enable 9 -> 9
    h0[14, 15] = -1.0  # Stop copying 14 -> 15
    h0[14, 16] = -1.0  # Stop copying 14 -> 16
    h0[15, 15] = 1.0  # Enable 15 -> 15
    h0[15, 16] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prevprev move, we enable feature 10 and 16
    h1[10, 10] = 1.0  # Enable 10 -> 10
    h1[15, 16] = -1.0  # Stop copying 15 -> 16
    h1[16, 16] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[13, 13] = 1.0

    # (1, n_bin, n_bin)
    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


def apply_history_matrices(model_config, batch_binaryInputNCL, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCL: (N, n_bin_in, L)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, L)
    batch_binaryInputNCL = torch.einsum("bcl,bcd->bdl", batch_binaryInputNCL, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCL, batch_globalInputNC
