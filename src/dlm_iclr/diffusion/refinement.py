"""CrysLLMGen refinement with retained single-request and explicit batched execution."""

from __future__ import annotations
import copy
from contextlib import nullcontext
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from dlm_iclr._core.dynamic_crystal import arrays_to_structure
from dlm_iclr._core.expert_edit_data import arrays_from_structure, quantize_arrays, certify_geometry
from dlm_iclr._core.continuous_keep_edit import geometry
from dlm_iclr._core.r03_physics_transfer import build_repair_constraints, geometry_support_report
from dlm_iclr.c1.generation import make_record
from dlm_iclr._core.paired_noise import derive_subseed


def refinement_noise(seeds, sizes, steps, device):
    """Independent per-request noise banks, unaffected by batching or worker order."""
    banks = {}
    for role in ("corrector_coordinates", "predictor_lattice", "predictor_coordinates"):
        values = []
        for seed, size in zip(seeds, sizes, strict=True):
            shape = (steps - 1, 1, 3, 3) if role == "predictor_lattice" else (steps - 1, size, 3)
            generator = torch.Generator(device=device).manual_seed(
                derive_subseed(int(seed), "refiner_request_noise_v1", role))
            values.append(torch.randn(shape, device=device, generator=generator))
        banks[role] = torch.cat(values, dim=1)
    return banks


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
    def __init__(self, checkpoint, tokenizer, device, *, steps=800, reuse_fixed_geometry=False,
                 reduction_protocol='legacy_scatter'):
        from dlm_iclr._vendor.crysllmgen.models_ddpm.diffusion import CSPDiffusion

        self.model = CSPDiffusion(1000, "train").to(device)
        self.model.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(saved["model"] if "model" in saved else saved, strict=True)
        self.model.eval()
        self.vocabulary = tokenizer.get_vocab()
        self.inverse = {int(value): token for token, value in self.vocabulary.items()}
        self.support = build_repair_constraints(tokenizer)
        self.steps = steps
        self.reuse_fixed_geometry = reuse_fixed_geometry
        if reduction_protocol not in ('legacy_scatter','ordered_csr_v1'):
            raise ValueError('Unknown refiner reduction protocol')
        self.reduction_protocol = reduction_protocol

    def reduction_context(self):
        if self.reduction_protocol=='ordered_csr_v1':
            from .reduction_audit import ordered_csp_reductions
            return ordered_csp_reductions()
        return nullcontext()

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
        with self.reduction_context():
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
            "reduction_protocol": self.reduction_protocol,
        }

    @torch.no_grad()
    def sample_many(self, graphs, seeds):
        """One GPU batch with independent, recorded per-request noise streams.

        This is a new batched execution protocol, not bitwise batch-one replay.
        Plan seeds determine noise independently of batch membership; batched
        matrix arithmetic can still differ numerically from separate calls.
        """
        from torch_geometric.data import Batch

        examples = [ProposalDataset(dict(graph, refiner_noise_seed=int(seed)))[0]
                    for graph, seed in zip(graphs, seeds, strict=True)]
        sizes = [int(example.num_atoms.item()) for example in examples]
        batch = Batch.from_data_list(examples).to(self.model.device)
        noise = refinement_noise(seeds, sizes, self.steps, self.model.device)
        with self.reduction_context():
            result, _ = self.model.sample(batch, diff_steps=self.steps,
                                          reuse_fixed_geometry=self.reuse_fixed_geometry,
                                          save_trajectory=False, noise_bank=noise)
        lengths, angles = lattices_to_parameters(result["lattices"])
        coordinates = result["frac_coords"].cpu().split(sizes)
        types = result["atom_types"].cpu().split(sizes)
        lattices, lengths, angles = result["lattices"].cpu(), lengths.cpu(), angles.cpu()
        values = []
        for i, seed in enumerate(seeds):
            if not all(bool(torch.isfinite(v).all()) for v in (coordinates[i], lattices[i], lengths[i], angles[i])):
                values.append({"error": "nonfinite_refiner_geometry"})
                continue
            values.append({
                "frac_coords": coordinates[i].tolist(), "atom_types": types[i].reshape(-1).tolist(),
                "lattice_matrix": lattices[i].tolist(), "lengths": lengths[i].tolist(),
                "angles": angles[i].tolist(), "seed": int(seed), "diffusion_steps": self.steps,
                "geometry_decoder_calls": 2 * self.steps,
                "execution": {"protocol": "refiner_request_noise_v1",
                              "reduction_protocol": self.reduction_protocol,
                              "batch_size": len(graphs), "batch_index": i},
            })
        return values

    def refine_many(self, plans, generated):
        active = [i for i, value in enumerate(generated) if value.get("graph") is not None]
        sampled = self.sample_many([generated[i]["graph"] for i in active],
                                   [plans[i]["refiner_noise_seed"] for i in active]) if active else []
        outputs = dict(zip(active, sampled, strict=True))
        return [self.refine(plan, value, sampled=outputs.get(i))
                for i, (plan, value) in enumerate(zip(plans, generated, strict=True))]

    def refine(self, plan, generated, *, sampled=None):
        from pymatgen.core import Lattice, Structure

        before = generated["record"]
        native, tokens = copy.deepcopy(before), list(before.get("body_token_ids") or [])
        trace, raw = {"source": "G", "editable": bool(tokens)}, None
        if generated.get("graph") is not None:
            try:
                raw = sampled if sampled is not None else self.sample(generated["graph"], plan["refiner_noise_seed"])
                if "error" in raw:
                    raise FloatingPointError(raw["error"])
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
