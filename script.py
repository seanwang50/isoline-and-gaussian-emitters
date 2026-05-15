"""
IsoLine and Gaussian Emitters (Sean Wang | May 2026)

Implementation script for article of matching title.

The default outputs are as follows:
The quality-diversity scores render in the terminal.
The corresponding heatmap comparisons store as PDFs.
"""

# The complete library dependencies.

from ribs.emitters import IsoLineEmitter
from ribs.emitters import GaussianEmitter

from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap

import numpy as np
from scipy import stats

import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# The global plot settings.

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],

        "font.size": 12,

        "axes.labelsize": 14,
        "axes.titlesize": 16,

        "axes.labelweight": "bold",
        "axes.titleweight": "bold",

        "figure.autolayout": False,
        "figure.constrained_layout.use": True,

        "savefig.format": "pdf",
        "savefig.bbox": "tight"
    }
)


# Our L-system settings.
PLAIN_RULES = {
        "X": [
            "F[+X][-X]", 
            "F[+X]", 
            "F[-X]"
        ],
        "F": [
            "FF",
            "F"
        ]
    }
AXIOM = "X"
ITERATION = 5


# The genome assembly function.

RuleBook = dict[str, list[tuple[float, str]]]

def assemble_genome(
    rulebook: RuleBook, 
    axiom: str, 
    iteration: int, 
    dice: np.random.Generator) -> str:
    """
    Assembles genome through recursive axiomatic transformations.
    
    Args:
        rulebook (RuleBook): The transformational mappings.
            - Format: {token: [(weight, option), ...], ...}
        axiom (str): The current substrate of transformations.
        iteration (int): The target number of transformations.
        dice (np.random.Generator): Isolated random number generator.
            - For thread safety during multi-core processing.

    Returns:
        genome (str): The final sequence of tokenized instructions.
    """
    if iteration == 0:
        genome = axiom
        return genome
    
    next_axiom = []

    for token in axiom:
        if token in rulebook:
            rules = rulebook[token]
            weights = [rule[0] for rule in rules]
            options = [rule[1] for rule in rules]

            transformation = dice.choice(a=options, p=weights)
            next_axiom.append(transformation)
        else:
            next_axiom.append(token)

    axiom = "".join(next_axiom)
    return assemble_genome(rulebook, axiom, iteration-1, dice)


# The branch assembly function

LineCoordinates = list[list[tuple[float, float]]]

def assemble_branches(genome: str) -> LineCoordinates:
    """
    Assembles branches by the below genomic interpretations.
        "F": Grow in the current orientation by 1.0 units.
        "X": Do Nothing.
        "+": Rotate the current orientation by 25.0 degrees counterclockwise.
        "-": Rotate the current orientation vy 25.0 degrees clockwise.
        "[": Save the current position
        "]": Return to the last saved position.

    Args:
        genome (str): The tokenized instruction for branch assembly.

    Returns:
        branches (LineCoordinates): The cartesian encoding of the branches.
            - Format: [[(initial_x, initial_y), (final_x, final_y)], ...]
    """
    branches = []
    holdings = []

    current_x = 0.0
    current_y = 0.0
    orientation = 90.0

    for token in genome:
        if token == "F":
            radian_orientation = np.radians(orientation)

            next_x = current_x + np.cos(radian_orientation)
            next_y = current_y + np.sin(radian_orientation)

            branches.append([(current_x, current_y), (next_x, next_y)])

            current_x = next_x
            current_y = next_y

        elif token == "+":
            orientation += 25.0
        elif token == "-":
            orientation -= 25.0

        elif token == "[":
            holdings.append((current_x, current_y, orientation))
        elif token == "]" and holdings:
            current_x, current_y, orientation = holdings.pop()
        else: continue

    return branches


# The branches evaluation function.

def evaluate_branches(branches: LineCoordinates) -> dict[str, float]:
    """
    Evaluates the given branches for the following measures.
        Structural Balance (Fitness): Normalized average of x-coordinates.
        Branch Density (Feature): The ratio between branch count and total area.
        Aspect Stature (Feature): The overall height-to-width ratio.

    Args:
        branches (LineCoordinates): The branches to be evaluated.
            - Format: [[(current_x, current_y), (next_x, next_y)], ...]

    Returns:
        measures (dict[str, float]): The labeled summary of measures.
            - Format: {measure_name: measure_value, ...}
    """
    if not branches:
        return{"balance": 0.0, "density": 0.0, "stature": 0.0}
    
    coordinates = np.array(branches)
    y_coordinates = coordinates[:, :, 1].flatten()
    x_coordinates = coordinates[:, :, 0].flatten()

    maximum_y, minimum_y = np.max(y_coordinates), np.min(y_coordinates)
    maximum_x, minimum_x = np.max(x_coordinates), np.min(x_coordinates)

    height = max(maximum_y - minimum_y, 1e-6)
    width  = max(maximum_x - minimum_x, 1e-6)

    average_x = np.mean(x_coordinates)
    balance = max(
        0.0, 1.0 - (abs(average_x) / (width / 2))
    ) if width > 0.0 else 0.0

    density = len(branches) / (height * width)
    stature = height / width

    measures = {
        "balance": float(balance),
        "density": float(density),
        "stature": float(stature)
    }
    return measures


# The weights evaluation function.

def evaluate_weights(
    weights: np.ndarray,
    seed: int, 
    trials: int=50) -> dict[str, float]:
    """
    Evaluates the given rule weights by averaging their
    resultant branch measures across trials.

    Args:
        weights (np.ndarray): The 1D array of rule weights.
            - Format: [weight_1, ..., weight_6].
        seed (int): Anchor for reproducible randomness.
        trials (int): The total number of measurement trials.
    
    Returns:
        average_measures (dict[str, float]): The labeled summary of scores.
            -Format = {measure_name: measure_value, ...}
    """
    dice = np.random.default_rng(seed)

    weights = np.clip(weights, 1e-6, 1.0)

    sum_X = np.sum(weights[0:3])
    weight_1 = weights[0] / sum_X
    weight_2 = weights[1] / sum_X
    weight_3 = weights[2] / sum_X

    sum_F = np.sum(weights[3:5])
    weight_4 = weights[3] / sum_F
    weight_5 = weights[4] / sum_F

    rulebook = {
        "X": [
            (weight_1, PLAIN_RULES["X"][0]),
            (weight_2, PLAIN_RULES["X"][1]),
            (weight_3, PLAIN_RULES["X"][2]),
        ],
        
        "F": [
            (weight_4, PLAIN_RULES["F"][0]),
            (weight_5, PLAIN_RULES["F"][1])
        ]
    }
    
    balance_sum = 0.0
    density_sum = 0.0
    stature_sum = 0.0

    for _ in range(trials):
        genome = assemble_genome(
            rulebook, 
            axiom=AXIOM, 
            iteration=ITERATION, 
            dice=dice
        )
        branches = assemble_branches(genome)
        scores = evaluate_branches(branches)

        balance_sum += scores["balance"]
        density_sum += scores["density"]
        stature_sum += scores["stature"]

    average_measures = {
        "balance": balance_sum / trials,
        "density": density_sum / trials,
        "stature": stature_sum / trials
    }
    return average_measures


# The archive illumination function.

def illuminate_archive(
    emitter_class: type, 
    emitter_kwargs: dict, 
    iterations: int, 
    seed: int) -> GridArchive:
    """
    Illuminates archive via the provided emitter type.

    Args:
        emitter_class (type): Either IsoLine or Gaussian emitter.
        emitter_kwargs (dict): Emitter-specific kwargs.
        iterations (int): The target number of search cycles.
        seed (int): Anchor for reproducible randomness.

    Returns:
        archive (GridArchive): The final archive.
    """
    archive = GridArchive(
        solution_dim=5,
        dims=[50, 50],
        ranges=[(0.0, 1.5), (0.0, 8.0)],
        seed=seed
    )

    initial_weights = np.array([0.34, 0.33, 0.33, 0.50, 0.50])
    weight_bounds = [(0.0, 1.0)] * 5
    
    emitters = [
        emitter_class(
            archive, 
            x0=initial_weights, 
            bounds=weight_bounds,
            batch_size=50,
            seed=seed,
            **emitter_kwargs
        )
    ]
    scheduler = Scheduler(archive, emitters)

    core_count = max(1, multiprocessing.cpu_count()-1)
    with multiprocessing.Pool(processes=core_count) as pool:
        for i in range(iterations):
            solutions = scheduler.ask()
            candidate_bits = [
                (solution, seed + (i * 10000) + j, 50) 
                for j, solution in enumerate(solutions)
            ]
            results = pool.starmap(evaluate_weights, candidate_bits)
            
            fitness_batch = []
            feature_batch = []
            
            for average_scores in results:
                density = average_scores["density"]
                stature = average_scores["stature"]
                
                if np.isfinite(density) and np.isfinite(stature):
                    fitness_batch.append(average_scores["balance"])
                    feature_batch.append([density, stature])
                else:
                    fitness_batch.append(-1.0)
                    feature_batch.append([0.0, 0.0])
                    
            scheduler.tell(fitness_batch, feature_batch)

    return archive


# The heatmap comparison function.

def save_comparison(
    subject_archive: GridArchive, 
    control_archive: GridArchive, 
    filename: str) -> None:
    """
    Plots the resultant archive of subject and control illuminations.
    The archives are displayed side-by-side.

    Args:
        subject_archive (GridArchive): Archive by subject illumination.
        control_archive (GridArchive): Archive by control illumination.
        filename (str): The destination file path/name for saving the figure.

    Returns:
        None. This function is for plotting only.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)


    minimum_fitness = 0.0
    maximum_fitness = 1.0

    grid_archive_heatmap(
        subject_archive,
        ax=ax1, 
        cmap="magma", 
        vmin=minimum_fitness, 
        vmax=maximum_fitness, 
        cbar=None
    )
    ax1.set_box_aspect(1)
    ax1.set_title(f"IsoLine")
    ax1.set_xlabel("Density")
    ax1.set_ylabel("Stature")
    ax1.tick_params(axis='both', which='major', pad=5)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    grid_archive_heatmap(
        control_archive,
        ax=ax2, 
        cmap="magma", 
        vmin=minimum_fitness, 
        vmax=maximum_fitness, 
        cbar=None
    )
    ax2.set_box_aspect(1)
    ax2.set_title(f"Gaussian")
    ax2.set_xlabel("Density")
    ax2.tick_params(axis='both', which='major', pad=5)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

    color_scalar = plt.cm.ScalarMappable(
        cmap="magma",
        norm=plt.Normalize(
            vmin=minimum_fitness, 
            vmax=maximum_fitness)
    )
    color_scalar.set_array([])

    colorbar = fig.colorbar(
        color_scalar,
        ax=[ax1, ax2],
        location='right',
        aspect=30
    )
    colorbar.set_label(
        "Balance",
        rotation=270,
        labelpad=15,
        weight="bold"
    )
    
    plt.savefig(filename, transparent=True)
    plt.close(fig)


# A side note on effect size.

def calculate_hedges_g(group_1: list[float], group_2: list[float]) -> float:
    """
    Calculates effect size as Hedges' g of the given pair of
    independent sample groups.

    Args:
        group_1 (list[float]): The 1st group of samples.
        group_2 (list[float]): The 2nd group of samples.

    Returns:
        hedges_g (float): Hedge's g as the effect size.
    """
    sample_size_1, sample_size_2 = len(group_1), len(group_2)

    variance_1, variance_2 = np.var(group_1, ddof=1), np.var(group_2, ddof=1)

    pooled_std = np.sqrt(
        ((sample_size_1 - 1) * variance_1 + (sample_size_2 - 1) * variance_2)
        / (sample_size_1 + sample_size_2 - 2)
    )
    cohens_d = (np.mean(group_1) - np.mean(group_2)) / pooled_std

    correction_factor = 1 - (3 / (4 * (sample_size_1 + sample_size_2) - 9))
    hedges_g = cohens_d * correction_factor

    return float(hedges_g)


# Putting everything together...

def run_experiment(trials: int, iterations: int) -> None:
    """
    Compares QD scores of subject and control illuminations across trials.

    Arguments:
        trials (int): The target number of comparisons.
        iterations (int): The target number of search cycles per illumination.
    """
    subject_qd_scores = []
    control_qd_scores = []
    
    print(f"Starting experiment:")
    print(f"{trials} trials, {iterations} iterations per illumination.\n")

    for i in range(trials):
        print(f"=== STARTING TRIAL {i+1}/{trials} (SEED {i}) ===")
        
        print("Illuminating subject...")
        subject_archive = illuminate_archive(
            emitter_class=IsoLineEmitter,
            emitter_kwargs={"iso_sigma": 0.1, "line_sigma": 0.2},
            iterations=iterations, 
            seed=i
        )
        subject_qd_score = subject_archive.stats.qd_score
        subject_qd_scores.append(subject_qd_score)
        
        print("Illuminating control...")
        control_archive = illuminate_archive(
            emitter_class=GaussianEmitter,
            emitter_kwargs={"sigma": 0.1},
            iterations=iterations, 
            seed=i
        )
        control_qd_score = control_archive.stats.qd_score
        control_qd_scores.append(control_qd_score)
        
        filename = f"comparison_trial_{i+1}_seed_{i}.pdf"
        save_comparison(subject_archive, control_archive, filename)
        
        print(
            f"Trial {i+1} Complete.",
            f"{'Subject (IsoLine)':<25} | {subject_qd_score:.1f}",
            f"{'Control (Gaussian)':<25} | {control_qd_score:.1f}",
            sep="\n"
        )
        print(f"Figure saved to:\n", f"{filename}\n", sep="")

    subject_qd_mean = np.mean(subject_qd_scores)
    control_qd_mean = np.mean(control_qd_scores)

    subject_qd_std = np.std(subject_qd_scores, ddof=1)
    control_qd_std = np.std(control_qd_scores, ddof=1)

    t_value, p_value = stats.ttest_ind(
        subject_qd_scores, control_qd_scores, equal_var=False
    )
    g_value= calculate_hedges_g(subject_qd_scores, control_qd_scores)

    print("=====================================================")
    print("               FINAL EXPERIMENT SUMMARY              ")
    print("=====================================================")
    print(f"{'Emitter Type':<25} | {'Mean QD Score ± Std'}")
    print("-" * 53)
    print(
        f"{'Subject (IsoLine)':<25} |",
        f"{subject_qd_mean:.2f} ± {subject_qd_std:.2f}"
    )
    print(
        f"{'Control (Gaussian)':<25} |",
        f"{control_qd_mean:.2f} ± {control_qd_std:.2f}"
    )
    print("-" * 53)
    print(f"{'Welch T-Value':<25} | {t_value:.4f}")
    print(f"{'Welch P-Value':<25} | {p_value:.4f}")
    print(f"{'Hedge G-Value':<25} | {g_value:.4f}")
    print("=====================================================")


# ==============================================================================

# Here we go!

# ==============================================================================


if __name__ == "__main__":

    run_experiment(trials=50, iterations=500)