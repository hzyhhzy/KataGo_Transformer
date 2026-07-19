import logging
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

import modelconfigs


def packed_full_board_rows(binary_input_nchw_packed: np.ndarray, pos_len: int) -> np.ndarray:
    """Return which rows have every on-board bit set in spatial feature 0."""
    if pos_len <= 0:
        raise ValueError(f"pos_len must be positive, got {pos_len}")
    if binary_input_nchw_packed.dtype != np.uint8:
        raise ValueError(
            "binaryInputNCHWPacked must have dtype uint8, got "
            f"{binary_input_nchw_packed.dtype}"
        )
    if binary_input_nchw_packed.ndim != 3 or binary_input_nchw_packed.shape[1] < 1:
        raise ValueError(
            "binaryInputNCHWPacked must have shape [N,C,packed_hw] with C >= 1, got "
            f"{binary_input_nchw_packed.shape}"
        )
    area = pos_len * pos_len
    full_bytes, remaining_bits = divmod(area, 8)
    required_bytes = full_bytes + (1 if remaining_bits else 0)
    if binary_input_nchw_packed.shape[2] != required_bytes:
        raise ValueError(
            f"Packed spatial input has {binary_input_nchw_packed.shape[2]} bytes, "
            f"but pos_len={pos_len} requires exactly {required_bytes}"
        )

    packed_mask = binary_input_nchw_packed[:, 0, :]
    rows = np.ones(packed_mask.shape[0], dtype=np.bool_)
    if full_bytes:
        rows &= np.all(packed_mask[:, :full_bytes] == np.uint8(0xFF), axis=1)
    if remaining_bits:
        high_bits_mask = np.uint8(((1 << remaining_bits) - 1) << (8 - remaining_bits))
        rows &= (packed_mask[:, full_bytes] & high_bits_mask) == high_bits_mask
    return rows

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
    require_full_board: bool = False,
    binary_input_nhwc: bool = False,
    filter_full_board_on_load: bool = False,
):
    if filter_full_board_on_load and not require_full_board:
        raise ValueError(
            "filter_full_board_on_load requires require_full_board=True"
        )

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

    include_qvalues = model_config["version"] >= 16 and model_config["version"] < 100

    def load_npz_file(npz_file):
        with np.load(npz_file) as npz:
            binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"]
            globalTargetsNC = npz["globalTargetsNC"]
            scoreDistrN = npz["scoreDistrN"]
            valueTargetsNCHW = npz["valueTargetsNCHW"]
            if include_meta:
                metadataInputNC = npz["metadataInputNC"]
            else:
                metadataInputNC = None
            if include_qvalues:
                qValueTargetsNCMove = npz["qValueTargetsNCMove"]
            else:
                qValueTargetsNCMove = None
        del npz

        if require_full_board:
            full_board_rows = packed_full_board_rows(binaryInputNCHWPacked, pos_len)
            if filter_full_board_on_load:
                source_num_samples = binaryInputNCHWPacked.shape[0]
                retained_num_samples = int(np.count_nonzero(full_board_rows))

                def filter_aligned_rows(name, array):
                    if array is None:
                        return None
                    if array.shape[0] != source_num_samples:
                        raise ValueError(
                            f"{npz_file} field {name} has {array.shape[0]} rows, "
                            f"expected {source_num_samples}"
                        )
                    if retained_num_samples == source_num_samples:
                        return array
                    return array[full_board_rows]

                binaryInputNCHWPacked = filter_aligned_rows(
                    "binaryInputNCHWPacked", binaryInputNCHWPacked
                )
                globalInputNC = filter_aligned_rows("globalInputNC", globalInputNC)
                policyTargetsNCMove = filter_aligned_rows(
                    "policyTargetsNCMove", policyTargetsNCMove
                )
                globalTargetsNC = filter_aligned_rows("globalTargetsNC", globalTargetsNC)
                scoreDistrN = filter_aligned_rows("scoreDistrN", scoreDistrN)
                valueTargetsNCHW = filter_aligned_rows(
                    "valueTargetsNCHW", valueTargetsNCHW
                )
                metadataInputNC = filter_aligned_rows(
                    "metadataInputNC", metadataInputNC
                )
                qValueTargetsNCMove = filter_aligned_rows(
                    "qValueTargetsNCMove", qValueTargetsNCMove
                )

                global_batch_size = batch_size * world_size
                if rank == 0 and retained_num_samples < global_batch_size:
                    logging.warning(
                        "%s full-board on-load filter retained %d/%d rows, fewer than "
                        "one global batch (%d = batch_size %d * world_size %d); "
                        "this file will yield no training batches",
                        npz_file,
                        retained_num_samples,
                        source_num_samples,
                        global_batch_size,
                        batch_size,
                        world_size,
                    )
            elif not np.all(full_board_rows):
                invalid_rows = np.flatnonzero(~full_board_rows)
                raise ValueError(
                    f"{npz_file} contains {invalid_rows.size} non-full-board rows "
                    f"(first row index {int(invalid_rows[0])}); -disable-mask is unsafe"
                )

        policyTargetsNCMove = policyTargetsNCMove.astype(np.float32)
        scoreDistrN = scoreDistrN.astype(np.float32)
        valueTargetsNCHW = valueTargetsNCHW.astype(np.float32)
        if metadataInputNC is not None:
            metadataInputNC = metadataInputNC.astype(np.float32)
        if qValueTargetsNCMove is not None:
            qValueTargetsNCMove = qValueTargetsNCMove.astype(np.float32)

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        return (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove)

    if not npz_files:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_npz_file, npz_files[0])

        for next_file in (npz_files[1:] + [None]):
            (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove) = future.result()

            num_samples = binaryInputNCHW.shape[0]
            # Just discard stuff that doesn't divide evenly
            num_whole_steps = num_samples // (batch_size * world_size)

            #logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            if next_file is not None:
                #logging.info(f"Preloading {next_file} while processing this file")
                future = executor.submit(load_npz_file, next_file)

            for n in range(num_whole_steps):
                start = (n * world_size + rank) * batch_size
                end = start + batch_size

                batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
                batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
                batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
                batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
                batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
                batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
                if include_qvalues:
                    batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)

                if enable_history_matrices:
                    (batch_binaryInputNCHW, batch_globalInputNC) = apply_history_matrices(
                        model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                    )
                if is_gomoku_history:
                    zero_mask = (torch.rand((batch_binaryInputNCHW.shape[0],), device=batch_binaryInputNCHW.device) < 0.3).to(batch_binaryInputNCHW.dtype)
                    batch_binaryInputNCHW[:, 6, :, :] *= (1.0 - zero_mask).view(-1, 1, 1)
                    batch_globalInputNC[:, 1] *= (1.0 - zero_mask)


                
                if symmetry_type is not None and symmetry_type!="" and symmetry_type!="none":
                    allowed_symms=[]
                    if symmetry_type == "xyt": # 8 symmetries,  Go, Gomoku ...
                        allowed_symms=[0,1,2,3,4,5,6,7]
                    elif symmetry_type == "x": # x-axis only, Chess-like
                        allowed_symms=[0,5]
                    elif symmetry_type == "xy": # x-axis or y-axis only
                        allowed_symms=[0,2,5,7]
                    elif symmetry_type == "x+y": # rotate 180 degrees. Hex
                        allowed_symms=[0,2]
                    elif symmetry_type == "t": # transpose only. Tiaoqi
                        allowed_symms=[0,4]
                    else:
                        assert False, f"Unknown data symmetry type {symmetry_type}"
                        
                    symm = allowed_symms[int(rand.integers(0,len(allowed_symms)))]
                    #logging.info(symm)
                               
                    batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                    batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)
                    batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
                    if include_qvalues:
                        batch_qValueTargetsNCMove = apply_symmetry_policy(batch_qValueTargetsNCMove, symm, pos_len)

                if binary_input_nhwc:
                    batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
                batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
                if include_qvalues:
                    batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()

                batch = dict(
                    binaryInputNCHW = batch_binaryInputNCHW,
                    globalInputNC = batch_globalInputNC,
                    policyTargetsNCMove = batch_policyTargetsNCMove,
                    globalTargetsNC = batch_globalTargetsNC,
                    scoreDistrN = batch_scoreDistrN,
                    valueTargetsNCHW = batch_valueTargetsNCHW,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC
                if include_qvalues:
                    batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove

                yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


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


def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCHW: (N, n_bin_in, 19, 19)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, 19, 19)
    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC
