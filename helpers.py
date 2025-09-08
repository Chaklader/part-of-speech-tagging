import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import networkx as nx
import random

from io import BytesIO
from itertools import chain
from collections import namedtuple, OrderedDict


"""
Named tuple representing a sentence with its words and corresponding POS tags.

This structure provides an immutable, lightweight container for sentence data
where words and tags are stored as parallel sequences, enabling efficient
access and iteration for sequence labeling tasks.

Attributes:
    words (tuple): Sequence of word tokens in the sentence
    tags (tuple): Corresponding POS tags for each word token
"""
Sentence = namedtuple("Sentence", "words tags")


def read_data(filename: str) -> OrderedDict:
    """
    Parse tagged sentence data from Brown corpus format file into structured format.
    
    Reads a corpus file containing sentences with word-tag pairs and converts it
    into an ordered dictionary of Sentence objects. The function handles the specific
    format where sentences are separated by blank lines and each sentence starts
    with a unique identifier followed by tab-separated word-tag pairs.
    
    File format expected:
    - Each sentence begins with a unique identifier on its own line
    - Following lines contain word[TAB]tag pairs
    - Sentences are separated by blank lines
    
    Args:
        filename (str): Path to the corpus file in Brown corpus format.
            File should contain sentences with unique identifiers and
            tab-separated word-tag pairs.
            
    Returns:
        OrderedDict: Dictionary mapping sentence identifiers to Sentence objects.
            Keys are sentence identifiers (str), values are Sentence namedtuples
            containing parallel sequences of words and tags.
            
    Example:
        >>> sentences = read_data("brown-corpus.txt")
        >>> sentence = sentences["b100-38532"]
        >>> print(sentence.words)
        ('Perhaps', 'it', 'was', 'right', ';', ';')
        >>> print(sentence.tags)
        ('ADV', 'PRON', 'VERB', 'ADJ', '.', '.')
        
    Note:
        Uses OrderedDict to preserve the original sentence ordering from the file,
        which can be important for maintaining chronological or topical structure
        in corpus analysis.
    """
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags(filename: str) -> frozenset:
    """
    Load the complete set of POS tags used in the corpus vocabulary.
    
    Reads a newline-separated file containing all possible POS tags and returns
    them as an immutable set. This tag vocabulary defines the complete label
    space for the part-of-speech tagging task.
    
    Args:
        filename (str): Path to file containing newline-separated POS tags.
            Each line should contain exactly one tag identifier.
            
    Returns:
        frozenset: Immutable set of all POS tag strings found in the file.
            Using frozenset ensures the tag vocabulary cannot be accidentally
            modified and provides efficient membership testing.
            
    Example:
        >>> tags = read_tags("tags-universal.txt")
        >>> print(sorted(tags))
        ['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']
        >>> 'NOUN' in tags
        True
        
    Note:
        The frozenset return type enables efficient set operations and ensures
        immutability, preventing accidental modification of the tag vocabulary
        during model training or evaluation.
    """
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)


def model2png(model, filename: str = "", overwrite: bool = False, show_ends: bool = False):
    """
    Convert a Pomegranate HMM model into a PNG visualization for analysis and presentation.
    
    This function creates a visual representation of the Hidden Markov Model's state
    structure and transitions using a multi-stage conversion pipeline. The resulting
    image shows states as nodes and transitions as directed edges, enabling intuitive
    understanding of the model's learned structure.
    
    The conversion process:
    1. Extracts the NetworkX graph from the Pomegranate model
    2. Converts to PyDot graph representation for layout optimization
    3. Renders to PNG using Graphviz dot layout algorithm
    4. Returns matplotlib-compatible image data
    
    Args:
        model: Pomegranate HMM model object with .graph attribute.
            Must contain a NetworkX graph representing model structure
            with states as nodes and transitions as edges.
        filename (str, optional): Output PNG file path. If provided, saves
            the visualization to disk. Defaults to empty string (no file output).
        overwrite (bool, optional): Whether to overwrite existing files.
            If False and filename exists, raises IOError. Defaults to False.
        show_ends (bool, optional): Whether to include start/end states in
            visualization. If False, hides implicit start/end states for cleaner
            display. Defaults to False.
            
    Returns:
        numpy.ndarray: PNG image data as matplotlib-compatible array.
            Can be displayed directly with plt.imshow() or saved separately.
            
    Raises:
        IOError: If filename exists and overwrite=False.
        AttributeError: If model lacks required .graph attribute.
        ImportError: If required visualization dependencies are missing.
        
    Example:
        >>> model = train_hmm_model(data)
        >>> img = model2png(model, "hmm_structure.png", overwrite=True)
        >>> plt.imshow(img)
        >>> plt.axis('off')
        >>> plt.show()
        
    Note:
        Requires PyDot and Graphviz for graph layout. The left-to-right
        layout (rankdir="LR") provides optimal visualization for sequence
        models like HMMs where temporal flow is important.
    """
    nodes = model.graph.nodes()
    if not show_ends:
        nodes = [n for n in nodes if n not in (model.start, model.end)]
    g = nx.relabel_nodes(model.graph.subgraph(nodes), {n: n.name for n in model.graph.nodes()})
    pydot_graph = nx.drawing.nx_pydot.to_pydot(g)
    pydot_graph.set_rankdir("LR")
    png_data = pydot_graph.create_png(prog='dot')
    img_data = BytesIO()
    img_data.write(png_data)
    img_data.seek(0)
    if filename:
        if os.path.exists(filename) and not overwrite:
            raise IOError("File already exists. Use overwrite=True to replace existing files on disk.")
        with open(filename, 'wb') as f:
            f.write(img_data.read())
        img_data.seek(0)
    return mplimg.imread(img_data)


def show_model(model, figsize: tuple = (5, 5), **kwargs):
    """
    Display a Pomegranate HMM model as an interactive matplotlib visualization.
    
    Convenience function that combines model2png conversion with matplotlib
    display setup. Creates a properly sized figure and displays the model
    structure without axis labels or ticks for clean presentation.
    
    This function is ideal for Jupyter notebook environments where immediate
    visual feedback is desired during model development and analysis.
    
    Args:
        model: Pomegranate HMM model object to visualize.
            Must have .graph attribute containing NetworkX representation.
        figsize (tuple, optional): Matplotlib figure dimensions as (width, height)
            in inches. Larger figures provide better detail for complex models.
            Defaults to (5, 5).
        **kwargs: Additional keyword arguments passed to model2png().
            Common options include show_ends, filename, and overwrite.
            
    Returns:
        None: Displays the model visualization using matplotlib.
        
    Example:
        >>> # Display simple model
        >>> show_model(trained_model)
        >>> 
        >>> # Display larger model with start/end states
        >>> show_model(complex_model, figsize=(10, 6), show_ends=True)
        >>> 
        >>> # Save while displaying
        >>> show_model(model, filename="model_viz.png", overwrite=True)
        
    Note:
        Uses plt.axis('off') to remove axis labels and ticks, focusing
        attention on the model structure. Figure size should be adjusted
        based on model complexity for optimal readability.
    """
    plt.figure(figsize=figsize)
    plt.imshow(model2png(model, **kwargs))
    plt.axis('off')


class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    """
    Immutable subset of corpus data optimized for machine learning workflows.
    
    This class provides a structured view of a subset of sentences from the corpus,
    pre-computing frequently needed derived data structures for efficient access
    during training and evaluation. It maintains both the original sentence structure
    and flattened sequences for different processing needs.
    
    The class extends namedtuple to provide immutability while adding computed
    properties and methods for common operations like iteration and length queries.
    
    Attributes:
        sentences (dict): Subset of sentences mapped by their identifiers
        keys (tuple): Ordered sequence of sentence identifiers in this subset
        vocab (frozenset): Complete vocabulary of words appearing in this subset
        X (tuple): Word sequences for all sentences (parallel to Y)
        tagset (frozenset): Complete set of POS tags used in this subset
        Y (tuple): Tag sequences for all sentences (parallel to X)
        N (int): Total number of word tokens across all sentences
        stream (callable): Iterator factory for (word, tag) pairs in corpus order
    """
    
    def __new__(cls, sentences: dict, keys: list):
        """
        Create a new Subset from selected sentences with pre-computed derived data.
        
        Constructs an immutable subset view that pre-computes vocabulary, tag sets,
        sequence data, and token counts for efficient access during model training
        and evaluation phases.
        
        Args:
            sentences (dict): Complete sentence collection mapped by identifiers
            keys (list): Sentence identifiers to include in this subset
            
        Returns:
            Subset: Immutable subset with pre-computed derived attributes
        """
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self) -> int:
        """Return the number of sentences in this subset."""
        return len(self.sentences)

    def __iter__(self):
        """Iterate over (sentence_id, sentence) pairs in this subset."""
        return iter(self.sentences.items())


class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    """
    Complete corpus dataset with automatic train-test splitting for POS tagging experiments.
    
    This class manages the entire corpus lifecycle from file loading through train-test
    splitting, providing structured access to both raw sentences and processed sequences.
    It pre-computes vocabulary statistics and maintains separate training and testing
    subsets for rigorous model evaluation.
    
    The class handles the complete preprocessing pipeline including file parsing,
    vocabulary extraction, sequence preparation, and randomized data splitting while
    maintaining reproducibility through configurable random seeds.
    
    Attributes:
        sentences (dict): Complete collection of sentences mapped by identifiers
        keys (tuple): All sentence identifiers in original order
        vocab (frozenset): Complete vocabulary across entire corpus
        X (tuple): All word sequences from the corpus
        tagset (frozenset): Complete POS tag set used in corpus
        Y (tuple): All tag sequences from the corpus  
        training_set (Subset): Training data subset (default 80% of sentences)
        testing_set (Subset): Testing data subset (default 20% of sentences)
        N (int): Total word tokens in entire corpus
        stream (callable): Iterator factory for all (word, tag) pairs
    """
    
    def __new__(cls, tagfile: str, datafile: str, train_test_split: float = 0.8, seed: int = 112890):
        """
        Initialize complete dataset with automatic preprocessing and train-test splitting.
        
        Loads corpus data from files, extracts vocabulary and tag sets, creates
        randomized train-test splits, and pre-computes derived data structures
        for efficient model training and evaluation.
        
        The splitting strategy uses randomization to ensure representative
        distribution of linguistic patterns across training and testing sets,
        while maintaining reproducibility through fixed random seeds.
        
        Args:
            tagfile (str): Path to file containing newline-separated POS tags.
                Should list all possible tags used in the corpus.
            datafile (str): Path to corpus file in Brown corpus format.
                Contains sentences with unique identifiers and word-tag pairs.
            train_test_split (float, optional): Fraction of data for training.
                Must be between 0 and 1. Defaults to 0.8 (80% training, 20% testing).
            seed (int, optional): Random seed for reproducible data splitting.
                Use None for non-deterministic splitting. Defaults to 112890.
                
        Returns:
            Dataset: Complete dataset with training and testing subsets.
            
        Example:
            >>> data = Dataset("tags-universal.txt", "brown-corpus.txt")
            >>> print(f"Training: {len(data.training_set)} sentences")
            >>> print(f"Testing: {len(data.testing_set)} sentences")
            >>> print(f"Vocabulary: {len(data.vocab)} words")
            >>> print(f"Tag set: {len(data.tagset)} tags")
            
        Note: 
            The random splitting ensures linguistic diversity in both sets but
            may break some topical or temporal coherence present in the original
            corpus ordering. Use fixed seeds for reproducible experiments.
        """
        tagset = read_tags(tagfile)
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        # split data into train/test sets
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self) -> int:
        """Return the total number of sentences in the complete dataset."""
        return len(self.sentences)

    def __iter__(self):
        """Iterate over all (sentence_id, sentence) pairs in the dataset."""
        return iter(self.sentences.items())