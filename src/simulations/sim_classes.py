import os
import sys
import yaml
import demes
import msprime
import subprocess
from pathlib import Path
from itertools import product


class BaseSimulator:
    """Abstract base class for all simulators."""

    def __init__(
        self,
        name: str,
        config_path: Path,
        sequence_length: float | None = None,
        population_size: float | None = None,
        mutation_rate: float | None = None,
        recombination_rate: float | None = None,
        num_replicates: float | None = None,
        save_trees: bool = False,
    ):
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.name = name
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.save_trees = save_trees

        # --- Load config ---
        self.graph = demes.load(config_path)

        # --- Set attributes, explicit parameters override config ---
        self.sequence_length = sequence_length if sequence_length is not None else graph.sequence_length
        self.population_size = population_size if population_size is not None else graph.population_size
        self.mutation_rate = mutation_rate if mutation_rate is not None else graph.mutation_rate
        self.recombination_rate = recombination_rate if recombination_rate is not None else graph.recombination_rate
        self.num_replicates = num_replicates if num_replicates is not None else 100

    def write_output(self, output_dir: Path, raw_output: str):
        """Writes {simname}_metadata.yaml and {simname}_output.txt files."""
        metadata = {
                    "simulator": self.name,
                    "sequence_length": self.sequence_length,
                    "population_size": self.population_size,
                    "num_replicates": self.num_replicates,
                    "overall_samples": self.overall_samples,
                    "has_recombination": bool(self.recombination_rate),
                    "config": str(self.config_path),
                }
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = output_dir / f"{self.name}_metadata.yaml"
        txt_path = output_dir / f"{self.name}_output.txt"

        # Convert Path objects in metadata to strings to avoid !!python/object issues
        safe_metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}

        with open(yaml_path, "w") as f_yaml:
            yaml.dump(safe_metadata, f_yaml, sort_keys=False)

        with open(txt_path, "w") as f_txt:
            f_txt.write(raw_output)

        print(f"✅ {self.name}: wrote {yaml_path.name} and {txt_path.name}")

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement `run()`")



# ─────────────────────────────────────────────
# Ms-like simulators (ms, msms, mspms)
# ─────────────────────────────────────────────
class MsLikeSimulator(BaseSimulator):
    """Parent class for ms-like simulators that require a population size."""

    def __init__(
        self,
        name: str,
        config_path: Path,
        sequence_length: float | None = None,
        population_size: float | None = None,
        mutation_rate: float | None = None,
        recombination_rate: float | None = None,
        num_replicates: float | None = None,
        save_trees: bool = False,
        path: str,
    ):
        super().__init__(name, sequence_length, population_size, mutation_rate, recombination_rate)
        self.path = path


class MsSimulator(MsLikeSimulator):
    """ms command-line simulator."""

    def run(self, config_path, output_dir, num_replicates, samples, overall_samples):
        graph = demes.load(config_path)
        cmd = demes.to_ms(
            graph,
            sequence_length=self.sequence_length,
            samples=samples,
            N0=self.population_size,
        )
        cmd = f"{self.path} {overall_samples} {num_replicates} {cmd} -T"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ms failed:\n{result.stderr}")

        metadata = {
            "simulator": self.name,
            "sequence_length": self.sequence_length,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "recombination_rate": self.recombination_rate,
            "num_replicates": num_replicates,
            "overall_samples": overall_samples,
            "config": Path(config_path).name,
        }
        self.write_output(output_dir, metadata, result.stdout)


# ─────────────────────────────────────────────
# msprime simulator
# ─────────────────────────────────────────────
class MsprimeSimulator(BaseSimulator):
    """Python-based msprime simulator (population_size optional)."""

    def __init__(
        self,
        sequence_length: float,
        population_size: float | None,
        mutation_rate: float,
        recombination_rate: float | None = None,
    ):
        super().__init__("msprime", sequence_length, population_size, mutation_rate, recombination_rate)

    def run(self, config_path, output_dir, num_replicates, samples, overall_samples):
        graph = demes.load(config_path)
        demography = msprime.Demography.from_demes(graph)

        sample_sets = [
            msprime.SampleSet(num_samples=s, population=pop["name"], time=0)
            for s, pop in zip(samples, graph.demes)
            if s > 0
        ]

        replicates = msprime.sim_ancestry(
            samples=sample_sets,
            demography=demography,
            num_replicates=num_replicates,
            sequence_length=self.sequence_length,
            recombination_rate=self.recombination_rate or 0,
        )

        output_lines = []
        for rep_index, ts in enumerate(replicates, start=1):
            mts = msprime.sim_mutations(
                ts,
                rate=self.mutation_rate,
                model=msprime.BinaryMutationModel(),
                discrete_genome=False,
            )
            output_lines.append(f"// Replicate {rep_index}")
            for tree in ts.trees():
                output_lines.append(tree.as_newick() + ";")
            output_lines.append(f"segsites: {mts.num_sites}")
            if mts.num_sites > 0:
                positions = " ".join(f"{site.position / mts.sequence_length:.6f}" for site in mts.sites())
                output_lines.append(f"positions: {positions}")
                for row in mts.genotype_matrix().T:
                    output_lines.append("".join(map(str, row)))
            output_lines.append("")

        metadata = {
            "simulator": self.name,
            "sequence_length": self.sequence_length,
            "population_size": self.population_size,  # can be None
            "mutation_rate": self.mutation_rate,
            "recombination_rate": self.recombination_rate,
            "num_replicates": num_replicates,
            "overall_samples": overall_samples,
            "config": Path(config_path).name,
        }

        self.write_output(output_dir, metadata, "\n".join(output_lines))


# ==========  Simulation Runner  ==========

class SimulationRunner:
    def __init__(self, simulators):
        self.simulators = simulators

    def parse_variables(self, filepath):
        """Parse variable file with lines like 'sample_size: 10,20,30'."""
        variables = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, values = line.split(":")
                key = key.strip()
                vals = [v.strip() for v in values.split(",")]
                def convert(v):
                    try:
                        return int(v) if "." not in v else float(v)
                    except ValueError:
                        return v
                variables[key] = [convert(v) for v in vals]
        return variables

    def run_simulations(self, config_path, output_dir, variable_file=None, reps=1):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        graph = demes.load(config_path)

        # Extract default parameters
        sequence_length = graph.sequence_length
        population_size = graph.population_size
        sample_size = [pop.get("sample_size", 1) for pop in config.get("demes")]

        combinations = [{}]
        if variable_file:
            vars_dict = self.parse_variables(variable_file)
            keys = list(vars_dict.keys())
            combinations = [dict(zip(keys, vals)) for vals in product(*vars_dict.values())]

        for combo in combinations:
            combo_dir = Path(output_dir) / "_".join(f"{k}-{v}" for k, v in combo.items()) if combo else Path(output_dir)
            combo_dir.mkdir(parents=True, exist_ok=True)

            for rep in range(reps):
                rep_dir = combo_dir / f"rep{rep+1}"
                rep_dir.mkdir(exist_ok=True)
                output_base = rep_dir / "sim"

                for sim in self.simulators:
                    sim.run(
                        config_path=config_path,
                        output_base=str(output_base) + f"_{sim.name}",
                        sequence_length=combo.get("sequence_length", sequence_length),
                        population_size=combo.get("population_size", population_size),
                        num_replicates=combo.get("num_replicates", 1),
                        sample_size=combo.get("sample_size", sample_size),
                        graph=graph,
                        config=config,
                    )
