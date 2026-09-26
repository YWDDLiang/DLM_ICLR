"""The CrysLLMGen model494 refinement, retaining batch-one noise semantics."""

from __future__ import annotations
import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from dlm_iclr._core.dynamic_crystal import arrays_to_structure
from dlm_iclr._core.expert_edit_data import arrays_from_structure, quantize_arrays, certify_geometry
from dlm_iclr._core.continuous_keep_edit import geometry
from dlm_iclr._core.geometry_constraints import build_repair_constraints, geometry_support_report
from dlm_iclr.periodic.generation import make_record


def graph_from_arrays(arrays):
    from dlm_iclr._vendor.crysllmgen.data_utils import process_one

    seen = set()
    for coord in arrays["frac_coords"]:
        key = tuple(int(round(float(value) * 100)) % 100 for value in coord)
        if key in seen:
            raise ValueError(f"duplicate periodic coordinate: {key}")
        seen.add(key)
    cif = arrays_to_structure(arrays).to(fmt="cif")
    *_, graph = process_one(cif, True, False, "crystalnn", False, 0.01)
    return graph, cif


class ProposalDataset(Dataset):
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return 1

    def __getitem__(self, index):
        from torch_geometric.data import Data

        graph = self.graph
        n = int(torch.as_tensor(graph["n_atom"]).view(-1)[0])
        return Data(
            num_atoms=torch.LongTensor([n]),
            num_nodes=n,
            num_bonds=graph["edge_indices"].shape[0],
            lengths=torch.as_tensor(graph["length"], dtype=torch.float32).view(1, 3),
            angles=torch.as_tensor(graph["angle"], dtype=torch.float32).view(1, 3),
            frac_coords=torch.as_tensor(graph["x_coord"], dtype=torch.float32),
            atom_types=torch.LongTensor(graph["a_type"]),
            edge_index=torch.LongTensor(graph["edge_indices"].T).contiguous(),
            to_jimages=torch.LongTensor(graph["to_jimages"]),
            sample_idx=torch.LongTensor([int(graph.get("sample_idx", index))]),
            refiner_seed=torch.LongTensor([graph["refiner_noise_seed"]]),
        )


def lattices_to_parameters(lattices):
    lengths = torch.sqrt(torch.sum(lattices**2, dim=-1))
    angles = torch.zeros_like(lengths)
    for index in range(3):
        j, k = (index + 1) % 3, (index + 2) % 3
        cosine = torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1) / (
            lengths[..., j] * lengths[..., k]
        )
        angles[..., index] = torch.clamp(cosine, -1.0, 1.0)
    return lengths, torch.arccos(angles) * 180.0 / np.pi


class Refiner:
    def __init__(self, checkpoint, tokenizer, device, *, steps=800, reuse_fixed_geometry=False):
        from dlm_iclr._vendor.crysllmgen.models_ddpm.diffusion import CSPDiffusion

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved["model"] if "model" in saved else saved
        timesteps = len(state["beta_scheduler.betas"]) - 1
        self.model = CSPDiffusion(timesteps, "train").to(device)
        self.model.device = torch.device(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.vocabulary = tokenizer.get_vocab()
        self.inverse = {int(value): token for token, value in self.vocabulary.items()}
        self.support = build_repair_constraints(tokenizer)
        self.steps = steps
        self.reuse_fixed_geometry = reuse_fixed_geometry

    @torch.no_grad()
    def sample(self, graph, seed):
        try:
            from torch_geometric.loader import DataLoader
        except ImportError:
            from torch_geometric.data import DataLoader

        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # DataLoader iterator creation follows reseeding, exactly as in the
        # original request loop. Moving it changes the consumed RNG stream.
        batch = next(
            iter(
                DataLoader(ProposalDataset(dict(graph, refiner_noise_seed=seed)), batch_size=1, shuffle=False)
            )
        ).to(self.model.device)
        options = {"reuse_fixed_geometry": True} if self.reuse_fixed_geometry else {}
        result, _ = self.model.sample(batch, diff_steps=self.steps, **options)
        lengths, angles = lattices_to_parameters(result["lattices"])
        if not all(
            bool(torch.isfinite(value).all())
            for value in (result["frac_coords"], result["lattices"], lengths, angles)
        ):
            raise FloatingPointError("nonfinite_refiner_geometry")
        return {
            "frac_coords": result["frac_coords"].cpu().tolist(),
            "atom_types": result["atom_types"].cpu().reshape(-1).tolist(),
            "lattice_matrix": result["lattices"].cpu().reshape(3, 3).tolist(),
            "lengths": lengths.cpu().reshape(3).tolist(),
            "angles": angles.cpu().reshape(3).tolist(),
            "seed": seed,
            "diffusion_steps": self.steps,
            "geometry_decoder_calls": 2 * self.steps,
        }

    def refine(self, plan, generated):
        from pymatgen.core import Lattice, Structure

        before = generated["record"]
        native, tokens = copy.deepcopy(before), list(before.get("body_token_ids") or [])
        trace, raw = {"source": "G", "editable": bool(tokens)}, None
        if generated.get("graph") is not None:
            try:
                raw = self.sample(generated["graph"], plan["refiner_noise_seed"])
                # The canonical lattice readout is part of the retained F endpoint.
                structure = Structure(
                    Lattice.from_parameters(*raw["lengths"], *raw["angles"]),
                    raw["atom_types"],
                    raw["frac_coords"],
                ).as_dict()
                refined = make_record(plan, stage="F", structure=structure)
                check = geometry(refined)
                if check["valid"] is True:
                    native, tokens = refined, []
                    trace = {"source": "continuous_F", "editable": False, "geometry": check}
                    try:
                        ids, decoded, diagnostic = quantize_arrays(
                            arrays_from_structure(structure), self.vocabulary
                        )
                        supported = geometry_support_report(ids, constraints=self.support)
                        if supported["supported"] and certify_geometry(decoded)["valid"] is True:
                            tokens = ids
                            trace.update(editable=True, quantization=diagnostic)
                        else:
                            trace["reason"] = "continuous_valid_without_supported_token_view"
                    except ValueError as error:
                        trace["reason"] = str(error)
                else:
                    trace.update(source="G_after_invalid_F_geometry", reason=check["reason"], geometry=check)
            except (ValueError, FloatingPointError) as error:
                trace.update(source="G_after_F_failure", reason=str(error))
        else:
            trace["reason"] = "no_construction_graph"
        return {"record": native, "token_ids": tokens, "continuous_trace": trace, "raw_refiner_output": raw}
