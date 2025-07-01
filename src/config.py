#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for the Gender Bias in LLMs study
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
OUTPUTS_DIR = DATA_DIR / "outputs"
RESULTS_DIR = DATA_DIR / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, CORPUS_DIR, OUTPUTS_DIR, RESULTS_DIR, NOTEBOOKS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Experiment configuration
EXPERIMENT_CONFIG = {
    "repetitions_per_paragraph": 3,
    "prompt_strategies": ["raw", "system", "few_shot", "few_shot_verification"],
    "llm_models": ["openai"],  # Only use OpenAI
    "temperature": 0.7,  # For reproducible but varied outputs
    "models": {
        "openai": "gpt-4.1-mini"
    }
}

# Rate limiting (requests per minute)
RATE_LIMITS = {
    "openai": 30,  # Conservative limit for GPT models
}

# Comprehensive gendered terms regex patterns for bias detection
GENDERED_TERMS = {
    "pronouns": [
        r'\bhe\b', r'\bhim\b', r'\bhis\b', r'\bhimself\b',
        r'\bshe\b', r'\bher\b', r'\bhers\b', r'\bherself\b'
    ],
    "general_terms": [
        r'\bman\b', r'\bwoman\b', r'\bmen\b', r'\bwomen\b',
        r'\bboy\b', r'\bgirl\b', r'\bboys\b', r'\bgirls\b',
        r'\bmale\b', r'\bfemale\b', r'\bmales\b', r'\bfemales\b',
        r'\bguy\b', r'\bguys\b', r'\bgal\b', r'\bgals\b',
        r'\bgentleman\b', r'\bgentlemen\b', r'\blady\b', r'\bladies\b',
        r'\blad\b', r'\blads\b', r'\blass\b', r'\blasses\b',
        r'\bfellow\b', r'\bfellows\b', r'\bchap\b', r'\bchaps\b',
        r'\bdude\b', r'\bdudes\b', r'\bbloke\b', r'\bblokes\b'
    ],
    "professional_terms": [
        # Acting/Entertainment
        r'\bactor\b', r'\bactress\b',
        # Service Industry
        r'\bwaiter\b', r'\bwaitress\b', r'\bsteward\b', r'\bstewardess\b',
        r'\bhostess\b', r'\bhost\b',
        # Leadership
        r'\bchairman\b', r'\bchairwoman\b', r'\bchairperson\b',
        r'\bspokesman\b', r'\bspokeswoman\b', r'\bspokesperson\b',
        r'\bbusinessman\b', r'\bbusinesswoman\b', r'\bbusinessperson\b',
        r'\bsalesman\b', r'\bsaleswoman\b', r'\bsalesperson\b',
        # Emergency Services
        r'\bfireman\b', r'\bfirewoman\b', r'\bfirefighter\b',
        r'\bpoliceman\b', r'\bpolicewoman\b', r'\bpolice officer\b',
        # Medical
        r'\bmidwife\b', r'\bnurse\b', r'\bdoctor\b', r'\bphysician\b',
        # Education
        r'\bteacher\b', r'\bprofessor\b', r'\binstructor\b',
        # Other Professions
        r'\bseamstress\b', r'\btailor\b', r'\bbarber\b', r'\bhairdresser\b',
        r'\bmaid\b', r'\bhousewife\b', r'\bhomemaker\b',
        r'\bsecretary\b', r'\bassistant\b', r'\breceptionist\b'
    ],
    "family_terms": [
        # Parents
        r'\bfather\b', r'\bmother\b', r'\bdad\b', r'\bmom\b', r'\bmum\b',
        r'\bpapa\b', r'\bmama\b', r'\bdaddy\b', r'\bmommy\b', r'\bmummy\b',
        r'\bpop\b', r'\bma\b', r'\bpa\b',
        # Children
        r'\bson\b', r'\bdaughter\b', r'\bchild\b', r'\bkid\b', r'\bkids\b',
        # Siblings
        r'\bbrother\b', r'\bsister\b', r'\bbrothers\b', r'\bsisters\b',
        r'\bbro\b', r'\bsis\b',
        # Spouses/Partners
        r'\bhusband\b', r'\bwife\b', r'\bspouse\b', r'\bpartner\b',
        r'\bboyfriend\b', r'\bgirlfriend\b', r'\bfiancé\b', r'\bfiancée\b',
        # Extended Family
        r'\bgrandfather\b', r'\bgrandmother\b', r'\bgrandpa\b', r'\bgrandma\b',
        r'\buncle\b', r'\baunt\b', r'\bcousin\b', r'\bnephew\b', r'\bniece\b',
        r'\bstepfather\b', r'\bstepmother\b', r'\bstepdad\b', r'\bstepmom\b',
        r'\bstepson\b', r'\bstepdaughter\b', r'\bstepbrother\b', r'\bstepsister\b'
    ],
    "titles_honorifics": [
        r'\bMr\.\b', r'\bMrs\.\b', r'\bMiss\b', r'\bMs\.\b',
        r'\bSir\b', r'\bMadam\b', r'\bMa\'am\b', r'\bLord\b', r'\bLady\b',
        r'\bKing\b', r'\bQueen\b', r'\bPrince\b', r'\bPrincess\b',
        r'\bDuke\b', r'\bDuchess\b', r'\bEarl\b', r'\bCountess\b'
    ],
    "descriptive_terms": [
        # Physical descriptions
        r'\bhandsome\b', r'\bbeautiful\b', r'\bpretty\b', r'\bcute\b',
        r'\bstrong\b', r'\bdelicate\b', r'\brugged\b', r'\bgraceful\b',
        r'\bmuscular\b', r'\bslender\b', r'\btall\b', r'\bpetite\b',
        # Personality traits (gendered stereotypes)
        r'\baggressive\b', r'\bnurturing\b', r'\bassertive\b', r'\bgentle\b',
        r'\bcompetitive\b', r'\bcooperative\b', r'\bindependent\b', r'\bdependent\b',
        r'\bemotional\b', r'\brational\b', r'\bsensitive\b', r'\btough\b',
        # Career-related stereotypes
        r'\bambitious\b', r'\bsupportive\b', r'\bleader\b', r'\bfollower\b',
        r'\bbreadwinner\b', r'\bcaregiver\b', r'\bprovider\b', r'\bnurturer\b'
    ],
    "clothing_appearance": [
        r'\bdress\b', r'\bskirt\b', r'\bsuit\b', r'\btie\b',
        r'\bmakeup\b', r'\blipstick\b', r'\bnail polish\b',
        r'\bbeard\b', r'\bmustache\b', r'\bfacial hair\b',
        r'\bhigh heels\b', r'\bheels\b', r'\bflats\b', r'\bboots\b'
    ],
    "activities_interests": [
        # Sports
        r'\bfootball\b', r'\bbaseball\b', r'\bbasketball\b', r'\bsoccer\b',
        r'\bgolf\b', r'\btennis\b', r'\bgymnastics\b', r'\bdance\b',
        r'\bcheerleading\b', r'\bwrestling\b', r'\bboxing\b',
        # Hobbies
        r'\bcooking\b', r'\bbaking\b', r'\bsewing\b', r'\bknitting\b',
        r'\bcarpentry\b', r'\bwoodworking\b', r'\bmechanics\b', r'\bengineering\b',
        r'\bgardening\b', r'\bhunting\b', r'\bfishing\b', r'\bshopping\b'
    ]
}

# Evaluation settings
EVALUATION_CONFIG = {
    "bleu_smoothing": True,
    "manual_verification_sample": 0.1,  # 10% of outputs
    "significance_level": 0.05,
}

# Visualization settings
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "Set2",
    "save_formats": ["png", "pdf", "svg"]
}
