"""
Curated dataset generators for geometric and topological experiments.

Each generator produces text datasets designed to exhibit specific
geometric or semantic properties when embedded, enabling controlled
experiments on embedding space structure.
"""

from __future__ import annotations

import logging
import random
from itertools import product

import numpy as np

from topo_llm.types import DatasetInfo

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate curated text datasets for embedding space experiments.

    All methods are static — no state is needed. Each returns texts
    and associated labels or scores for supervised analysis.

    Examples
    --------
    >>> texts, labels = DatasetGenerator.semantic_categories(n_per_category=50)
    >>> len(texts)
    500
    >>> set(labels)
    {'animals', 'vehicles', 'emotions', ...}
    """

    # ── Semantic category data ────────────────────────────────

    CATEGORIES: dict[str, list[str]] = {
        "animals": [
            "dog", "cat", "elephant", "tiger", "penguin", "dolphin", "eagle",
            "snake", "rabbit", "bear", "wolf", "lion", "giraffe", "whale",
            "octopus", "parrot", "horse", "deer", "shark", "butterfly",
            "cheetah", "gorilla", "kangaroo", "panda", "flamingo", "owl",
            "turtle", "fox", "seal", "bat",
        ],
        "vehicles": [
            "car", "bicycle", "airplane", "helicopter", "submarine", "train",
            "motorcycle", "bus", "truck", "sailboat", "canoe", "spaceship",
            "tractor", "ambulance", "taxi", "ferry", "scooter", "jet ski",
            "gondola", "hovercraft", "rickshaw", "trolley", "tank", "yacht",
            "skateboard", "segway", "blimp", "sled", "kayak", "raft",
        ],
        "emotions": [
            "happiness", "sadness", "anger", "fear", "surprise", "disgust",
            "love", "jealousy", "pride", "shame", "guilt", "anxiety",
            "excitement", "boredom", "nostalgia", "hope", "despair",
            "contentment", "frustration", "awe", "gratitude", "loneliness",
            "curiosity", "empathy", "serenity", "rage", "melancholy",
            "euphoria", "dread", "compassion",
        ],
        "colors": [
            "red", "blue", "green", "yellow", "purple", "orange", "pink",
            "black", "white", "gray", "brown", "cyan", "magenta", "turquoise",
            "indigo", "violet", "crimson", "scarlet", "navy", "olive",
            "maroon", "teal", "coral", "gold", "silver", "ivory", "beige",
            "amber", "lavender", "periwinkle",
        ],
        "countries": [
            "France", "Japan", "Brazil", "Egypt", "Canada", "Australia",
            "India", "Germany", "Mexico", "Nigeria", "Sweden", "Argentina",
            "Thailand", "Morocco", "Peru", "Greece", "Vietnam", "Kenya",
            "Ireland", "Chile", "Poland", "Indonesia", "Turkey", "Colombia",
            "Norway", "Portugal", "Malaysia", "Ethiopia", "Cuba", "Finland",
        ],
        "mathematical_concepts": [
            "derivative", "integral", "matrix", "vector", "topology",
            "probability", "eigenvalue", "manifold", "group theory",
            "differential equation", "Fourier transform", "prime number",
            "complex analysis", "linear algebra", "graph theory",
            "combinatorics", "set theory", "calculus", "geometry",
            "statistics", "number theory", "abstract algebra",
            "optimization", "tensor", "symmetry", "chaos theory",
            "fractal", "Bayesian inference", "Markov chain", "homotopy",
        ],
        "musical_instruments": [
            "piano", "guitar", "violin", "drums", "flute", "trumpet",
            "cello", "saxophone", "clarinet", "harp", "trombone", "oboe",
            "accordion", "banjo", "ukulele", "sitar", "harmonica",
            "xylophone", "bagpipes", "organ", "bass guitar", "mandolin",
            "tuba", "french horn", "didgeridoo", "tabla", "maracas",
            "bongo", "timpani", "synthesizer",
        ],
        "diseases": [
            "diabetes", "malaria", "influenza", "tuberculosis", "cancer",
            "asthma", "arthritis", "Alzheimer's", "Parkinson's", "epilepsy",
            "hepatitis", "pneumonia", "measles", "cholera", "anemia",
            "leukemia", "migraine", "hypertension", "osteoporosis",
            "bronchitis", "meningitis", "dengue", "eczema", "gout",
            "fibromyalgia", "lupus", "sclerosis", "psoriasis",
            "glaucoma", "sepsis",
        ],
        "programming_languages": [
            "Python", "JavaScript", "Java", "C++", "Rust", "Go",
            "TypeScript", "Swift", "Kotlin", "Ruby", "PHP", "Scala",
            "Haskell", "Perl", "Lua", "R", "MATLAB", "Julia",
            "Elixir", "Clojure", "Dart", "Fortran", "COBOL",
            "Assembly", "Prolog", "Erlang", "OCaml", "Zig",
            "Nim", "Crystal",
        ],
        "foods": [
            "pizza", "sushi", "pasta", "tacos", "curry", "hamburger",
            "croissant", "dim sum", "paella", "ramen", "falafel",
            "biryani", "goulash", "pho", "ceviche", "risotto",
            "kimchi", "hummus", "tiramisu", "pad thai", "empanada",
            "borscht", "gyoza", "couscous", "baklava", "gnocchi",
            "bibimbap", "moussaka", "pierogi", "shakshuka",
        ],
    }

    TEMPLATES: list[str] = [
        "A {item} is a type of {category}.",
        "{item} is commonly associated with {category}.",
        "When people think of {category}, {item} often comes to mind.",
        "In the domain of {category}, {item} is a well-known example.",
        "{item} is classified under the category of {category}.",
        "Among various {category}, {item} is particularly notable.",
        "The concept of {item} belongs to the field of {category}.",
        "One prominent example of {category} is {item}.",
        "{item} serves as a representative instance of {category}.",
        "Within {category}, {item} holds a significant place.",
        "Researchers studying {category} frequently encounter {item}.",
        "{item} exemplifies the characteristics typical of {category}.",
    ]

    @staticmethod
    def semantic_categories(
        n_per_category: int = 100,
        seed: int = 42,
    ) -> tuple[list[str], list[str], DatasetInfo]:
        """Generate texts belonging to distinct semantic categories.

        For each of 10 categories, generates ``n_per_category`` short
        descriptive sentences using varied templates.

        Parameters
        ----------
        n_per_category : int
            Number of texts to generate per category.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        texts : list[str]
            Generated text samples.
        labels : list[str]
            Category label for each text.
        info : DatasetInfo
            Dataset metadata.
        """
        rng = random.Random(seed)
        texts: list[str] = []
        labels: list[str] = []

        category_names = sorted(DatasetGenerator.CATEGORIES.keys())

        for category in category_names:
            items = DatasetGenerator.CATEGORIES[category]
            # Normalize category name for templates
            cat_display = category.replace("_", " ")

            for i in range(n_per_category):
                item = items[i % len(items)]
                template = DatasetGenerator.TEMPLATES[i % len(DatasetGenerator.TEMPLATES)]
                text = template.format(item=item, category=cat_display)
                texts.append(text)
                labels.append(category)

        # Shuffle together
        combined = list(zip(texts, labels))
        rng.shuffle(combined)
        texts, labels = zip(*combined)  # type: ignore[assignment]

        info = DatasetInfo(
            name="semantic_categories",
            n_samples=len(texts),
            n_categories=len(category_names),
            category_names=category_names,
            description=f"10 semantic categories, {n_per_category} texts each",
        )

        return list(texts), list(labels), info

    # ── Factual vs. fabricated ────────────────────────────────

    FACTUAL_PAIRS: list[tuple[str, str, str]] = [
        # (domain, factual, fabricated)
        # Geography
        ("geography", "The capital of France is Paris.", "The capital of France is Madrid."),
        ("geography", "The Amazon River flows through Brazil.", "The Amazon River flows through Argentina."),
        ("geography", "Mount Everest is the tallest mountain on Earth.", "Mount Kilimanjaro is the tallest mountain on Earth."),
        ("geography", "The Sahara Desert is located in Africa.", "The Sahara Desert is located in Asia."),
        ("geography", "Tokyo is the capital of Japan.", "Osaka is the capital of Japan."),
        ("geography", "The Nile is the longest river in Africa.", "The Congo is the longest river in Africa."),
        ("geography", "Australia is both a country and a continent.", "Australia is a country but not a continent."),
        ("geography", "The Pacific Ocean is the largest ocean.", "The Atlantic Ocean is the largest ocean."),
        ("geography", "Iceland is located in the North Atlantic.", "Iceland is located in the South Pacific."),
        ("geography", "The Great Wall of China is visible from the ground.", "The Great Wall of China is visible from the Moon."),
        ("geography", "Berlin is the capital of Germany.", "Munich is the capital of Germany."),
        ("geography", "The Ganges River flows through India.", "The Ganges River flows through Pakistan."),
        ("geography", "Canada has the longest coastline in the world.", "Russia has the longest coastline in the world."),
        ("geography", "The Dead Sea is the lowest point on land.", "The Caspian Sea is the lowest point on land."),
        ("geography", "Denmark is a Scandinavian country.", "Denmark is a Baltic country."),
        # Science
        ("science", "Water boils at 100 degrees Celsius at sea level.", "Water boils at 90 degrees Celsius at sea level."),
        ("science", "The speed of light is approximately 300,000 km/s.", "The speed of light is approximately 150,000 km/s."),
        ("science", "DNA has a double helix structure.", "DNA has a triple helix structure."),
        ("science", "Photosynthesis converts CO2 into oxygen.", "Photosynthesis converts oxygen into CO2."),
        ("science", "The human body has 206 bones.", "The human body has 186 bones."),
        ("science", "Electrons have a negative charge.", "Electrons have a positive charge."),
        ("science", "The Earth revolves around the Sun.", "The Sun revolves around the Earth."),
        ("science", "Gravity accelerates objects at 9.8 m/s² on Earth.", "Gravity accelerates objects at 12.5 m/s² on Earth."),
        ("science", "Sound travels faster in water than in air.", "Sound travels faster in air than in water."),
        ("science", "Diamonds are made of carbon atoms.", "Diamonds are made of silicon atoms."),
        ("science", "The mitochondria is the powerhouse of the cell.", "The nucleus is the powerhouse of the cell."),
        ("science", "Mercury is the closest planet to the Sun.", "Venus is the closest planet to the Sun."),
        ("science", "Helium is the second lightest element.", "Helium is the third lightest element."),
        ("science", "Antibiotics kill bacteria, not viruses.", "Antibiotics kill both bacteria and viruses equally."),
        ("science", "The pH of pure water is 7.", "The pH of pure water is 5."),
        # History
        ("history", "World War II ended in 1945.", "World War II ended in 1943."),
        ("history", "The French Revolution began in 1789.", "The French Revolution began in 1776."),
        ("history", "The printing press was invented by Gutenberg.", "The printing press was invented by Da Vinci."),
        ("history", "The Roman Empire fell in 476 AD.", "The Roman Empire fell in 576 AD."),
        ("history", "The Berlin Wall fell in 1989.", "The Berlin Wall fell in 1991."),
        ("history", "The first moon landing was in 1969.", "The first moon landing was in 1972."),
        ("history", "The Declaration of Independence was signed in 1776.", "The Declaration of Independence was signed in 1783."),
        ("history", "Napoleon was exiled to Elba.", "Napoleon was exiled to Corsica."),
        ("history", "The Titanic sank in 1912.", "The Titanic sank in 1915."),
        ("history", "Alexander the Great was from Macedonia.", "Alexander the Great was from Sparta."),
        ("history", "The Renaissance began in Italy.", "The Renaissance began in France."),
        ("history", "The Magna Carta was signed in 1215.", "The Magna Carta was signed in 1315."),
        ("history", "Columbus reached the Americas in 1492.", "Columbus reached the Americas in 1488."),
        ("history", "The Industrial Revolution started in Britain.", "The Industrial Revolution started in Germany."),
        ("history", "The Cold War ended with the dissolution of the USSR.", "The Cold War ended with the fall of the Berlin Wall alone."),
        # Mathematics
        ("mathematics", "Pi is approximately 3.14159.", "Pi is approximately 3.15259."),
        ("mathematics", "The square root of 144 is 12.", "The square root of 144 is 14."),
        ("mathematics", "A triangle has three sides.", "A triangle has four sides."),
        ("mathematics", "The sum of angles in a triangle is 180 degrees.", "The sum of angles in a triangle is 360 degrees."),
        ("mathematics", "Zero is an even number.", "Zero is an odd number."),
        ("mathematics", "The derivative of x² is 2x.", "The derivative of x² is x²."),
        ("mathematics", "There are infinitely many prime numbers.", "There are exactly one million prime numbers."),
        ("mathematics", "The Fibonacci sequence starts with 0, 1, 1, 2, 3.", "The Fibonacci sequence starts with 1, 2, 3, 5, 8."),
        ("mathematics", "Euler's number e is approximately 2.718.", "Euler's number e is approximately 2.618."),
        ("mathematics", "A circle's circumference is 2πr.", "A circle's circumference is πr²."),
    ]

    @staticmethod
    def factual_vs_fabricated(
        n_pairs: int = 200,
        seed: int = 42,
    ) -> tuple[list[str], list[bool], DatasetInfo]:
        """Generate paired factual and fabricated statements.

        Each pair consists of a true statement and a plausible but
        false variant. Covers geography, science, history, and mathematics.

        Parameters
        ----------
        n_pairs : int
            Target number of pairs. If more pairs than available data,
            the available pairs are repeated cyclically.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        texts : list[str]
            All texts (factual and fabricated interleaved).
        is_factual : list[bool]
            True if the corresponding text is factual.
        info : DatasetInfo
            Dataset metadata.
        """
        rng = random.Random(seed)

        all_pairs = DatasetGenerator.FACTUAL_PAIRS
        texts: list[str] = []
        is_factual: list[bool] = []

        for i in range(n_pairs):
            domain, factual, fabricated = all_pairs[i % len(all_pairs)]
            texts.append(factual)
            is_factual.append(True)
            texts.append(fabricated)
            is_factual.append(False)

        # Shuffle together
        combined = list(zip(texts, is_factual))
        rng.shuffle(combined)
        texts, is_factual = zip(*combined)  # type: ignore[assignment]

        info = DatasetInfo(
            name="factual_vs_fabricated",
            n_samples=len(texts),
            n_categories=2,
            category_names=["factual", "fabricated"],
            description=f"{n_pairs} factual/fabricated pairs across 4 domains",
        )

        return list(texts), list(is_factual), info

    # ── Graduated similarity ──────────────────────────────────

    SIMILARITY_ANCHORS: list[dict[str, str]] = [
        {
            "anchor": "The cat sat on the mat.",
            "1.0": "A feline rested on the rug.",
            "0.75": "The dog lay on the carpet.",
            "0.5": "The furniture was comfortable and soft.",
            "0.25": "Rain fell steadily on Tuesday afternoon.",
            "0.0": "Quantum entanglement violates Bell inequalities.",
        },
        {
            "anchor": "The stock market crashed dramatically yesterday.",
            "1.0": "Financial markets experienced a severe downturn yesterday.",
            "0.75": "The economy showed signs of significant weakness.",
            "0.5": "Investors were concerned about their portfolios.",
            "0.25": "The library was quiet in the afternoon.",
            "0.0": "Photosynthesis requires chlorophyll molecules.",
        },
        {
            "anchor": "She played a beautiful melody on the piano.",
            "1.0": "A lovely tune was performed on the keyboard.",
            "0.75": "The musician gave a wonderful concert performance.",
            "0.5": "Music can evoke deep emotional responses.",
            "0.25": "The highway was congested during rush hour.",
            "0.0": "Tectonic plates cause earthquakes at fault lines.",
        },
        {
            "anchor": "The scientist discovered a new species of butterfly.",
            "1.0": "A researcher identified an unknown butterfly species.",
            "0.75": "Biologists found rare insects in the rainforest.",
            "0.5": "Scientific research often reveals unexpected findings.",
            "0.25": "The meeting was scheduled for three o'clock.",
            "0.0": "Parliamentary democracy originated in England.",
        },
        {
            "anchor": "Children were playing soccer in the park.",
            "1.0": "Kids kicked a ball around on the field.",
            "0.75": "The youth basketball game was exciting to watch.",
            "0.5": "Outdoor activities are beneficial for health.",
            "0.25": "The server processed the database query slowly.",
            "0.0": "Magnesium has an atomic number of twelve.",
        },
        {
            "anchor": "The restaurant served excellent Italian food.",
            "1.0": "The eatery provided outstanding Mediterranean cuisine.",
            "0.75": "The French bistro had a wonderful dinner menu.",
            "0.5": "Dining out can be an enjoyable social experience.",
            "0.25": "The algorithm had a time complexity of O(n log n).",
            "0.0": "Glaciers have been retreating since the ice age.",
        },
        {
            "anchor": "The professor lectured about quantum mechanics.",
            "1.0": "The teacher gave a talk on quantum physics.",
            "0.75": "The seminar covered advanced topics in theoretical physics.",
            "0.5": "University education involves many complex subjects.",
            "0.25": "The garden needed watering after the dry spell.",
            "0.0": "Renaissance painters used egg tempera techniques.",
        },
        {
            "anchor": "Heavy rain caused flooding in the coastal town.",
            "1.0": "Severe rainfall led to floods in the seaside village.",
            "0.75": "The hurricane brought devastating storm surges.",
            "0.5": "Weather patterns affect communities in many ways.",
            "0.25": "She finished reading the novel before midnight.",
            "0.0": "Binary search trees enable efficient data retrieval.",
        },
        {
            "anchor": "The astronaut floated weightlessly inside the space station.",
            "1.0": "The cosmonaut drifted in zero gravity aboard the orbital lab.",
            "0.75": "Life in microgravity presents unique physical challenges.",
            "0.5": "Space exploration pushes the boundaries of human knowledge.",
            "0.25": "The bakery opened early every Saturday morning.",
            "0.0": "Impressionist painters preferred painting outdoors.",
        },
        {
            "anchor": "The surgeon performed a complex heart operation.",
            "1.0": "The doctor carried out an intricate cardiac procedure.",
            "0.75": "The medical team completed a challenging brain surgery.",
            "0.5": "Modern medicine has advanced significantly in recent decades.",
            "0.25": "The river flowed calmly through the valley.",
            "0.0": "Fibonacci numbers appear in many natural patterns.",
        },
    ]

    @staticmethod
    def graduated_similarity(
        n_anchors: int = 50,
        seed: int = 42,
    ) -> tuple[list[str], list[str], list[float], DatasetInfo]:
        """Generate anchor-comparison pairs at graduated similarity levels.

        For each anchor sentence, generates comparisons at five
        similarity levels: 0.0, 0.25, 0.5, 0.75, and 1.0.

        Parameters
        ----------
        n_anchors : int
            Number of anchor sentences. Cycles through available anchors.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        anchors : list[str]
            Anchor sentences (repeated for each similarity level).
        comparisons : list[str]
            Comparison sentences.
        similarity_scores : list[float]
            Similarity score for each (anchor, comparison) pair.
        info : DatasetInfo
            Dataset metadata.
        """
        rng = random.Random(seed)

        all_anchors = DatasetGenerator.SIMILARITY_ANCHORS
        anchors: list[str] = []
        comparisons: list[str] = []
        scores: list[float] = []
        levels = ["0.0", "0.25", "0.5", "0.75", "1.0"]

        for i in range(n_anchors):
            data = all_anchors[i % len(all_anchors)]
            for level in levels:
                anchors.append(data["anchor"])
                comparisons.append(data[level])
                scores.append(float(level))

        # Shuffle together
        combined = list(zip(anchors, comparisons, scores))
        rng.shuffle(combined)
        anchors, comparisons, scores = zip(*combined)  # type: ignore[assignment]

        info = DatasetInfo(
            name="graduated_similarity",
            n_samples=len(anchors),
            n_categories=5,
            category_names=["0.0", "0.25", "0.5", "0.75", "1.0"],
            description=f"{n_anchors} anchors x 5 similarity levels",
        )

        return list(anchors), list(comparisons), list(scores), info

    @staticmethod
    def load_truthfulqa(
        split: str = "validation",
    ) -> tuple[list[str], list[str], list[bool], DatasetInfo]:
        """Load TruthfulQA dataset from HuggingFace.

        Parameters
        ----------
        split : str
            Dataset split to load.

        Returns
        -------
        questions : list[str]
            Questions from the dataset.
        best_answers : list[str]
            Best correct answers.
        is_truthful : list[bool]
            Whether the best answer is truthful (always True for best_answer).
        info : DatasetInfo
            Dataset metadata.

        Raises
        ------
        ImportError
            If the ``datasets`` library is not installed.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for TruthfulQA. "
                "Install with: pip install datasets"
            )

        logger.info("Loading TruthfulQA dataset (split=%s)", split)
        ds = load_dataset("truthful_qa", "generation", split=split)

        questions: list[str] = []
        best_answers: list[str] = []
        is_truthful: list[bool] = []

        for row in ds:
            questions.append(row["question"])
            best_answers.append(row["best_answer"])
            is_truthful.append(True)  # best_answer is always truthful

        info = DatasetInfo(
            name="truthfulqa",
            n_samples=len(questions),
            description=f"TruthfulQA {split} split, {len(questions)} questions",
        )

        return questions, best_answers, is_truthful, info
