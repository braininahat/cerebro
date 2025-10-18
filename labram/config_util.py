import os


class Config:
    """Centralized configuration for Route-B pretraining"""

    # Data paths
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    CACHE_DIR = os.path.join(DATA_DIR, "cache")

    # Releases
    RELEASES = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
    TEST_RELEASE = "R5"

    CHALLENGE_TASK = "contrastChangeDetection"
    CHALLENGE_2_DESCRIPTION = "externalizing"
    # Passive tasks for pre-training
    PASSIVE_TASKS = [
        "restingState",
        "surroundSupp",
        "despicableMe",
        "thePresent",
        "diaryOfAWimpyKid",
        "funwithFractals",
    ]

    # Excluded subjects
    EXCLUDED_SUBJECTS = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
        "NDARBA381JGH"
    ]

    # Preprocessing
    SFREQ = 100
    L_FREQ = 0.5
    H_FREQ = 40.0
    MIN_DURATION_S = 4.0
    N_CHANNELS = 129

    # Windowing
    WINDOW_SIZE_S = 4.0
    WINDOW_STRIDE_S = 2.0
    CROP_SIZE_S = 2.0
    SHIFT_AFTER_STIM = 0.5
    INPUT_SIZE = int(SFREQ * CROP_SIZE_S)  # 200 samples

    # Patch spec
    PATCH_LEN = 20
    N_FFT_BINS = 5  # Changed to 5 frequency bands for band-based FFT
    USE_BAND_FFT = True  # Use frequency bands instead of raw FFT

    # Model sizes
    TOK_DIM = 256
    CODEBOOK_SIZE = 8192
    CODEBOOK_EMD_DIM = 64
    EMA_DECAY = 0.99
    VQ_TYPE = "gumbel"  # "gumbel" or "ema"
    TEMPERATURE = 1.0
    KL_WEIGHT = 0.01

    MC_D = 256
    MC_LAYERS = 8
    MC_HEADS = 8
    MASK_RATIO = 0.5
    MASK_STRATEGY = "spatial"  # "spatial", "channel", or "random"

    # Training
    BATCH_SIZE = 256
    NUM_WORKERS = 8
    MAX_EPOCHS = 100
    LR = 1e-3
    VAL_RATIO = 0.1

    # I/O
    ROOT_DIR = "./checkpoints/labram"
    TOK_DIR = os.path.join(ROOT_DIR, "tokenizer")
    MEM_DIR = os.path.join(ROOT_DIR, "mem_pretrain")
    FINETUNE_DIR = os.path.join(ROOT_DIR, "finetune")
    CODES_DIR = os.path.join(ROOT_DIR, "codes_passive")
    MC_DIR = os.path.join(ROOT_DIR, "masked")
    FT_DIR = os.path.join(ROOT_DIR, "finetune")

    LOG_DIR = "./logs/labram"

    # WandB
    WANDB_PROJECT = "eeg-challenge-2025"
    WANDB_ENTITY = None
    USE_WANDB = True

    # Seeds
    SEED = 42
