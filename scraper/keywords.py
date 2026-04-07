"""
Seed keyword lists for relevance pre-filtering.
Expand these as you build your lexicon in later project stages.
"""

CLIMATE_KEYWORDS = [
    # Weather events
    "monsoon", "rainfall", "flood", "flooding", "drought", "heat wave",
    "heatwave", "cold wave", "cyclone", "storm", "unseasonal rain",
    "deficit rainfall", "excess rainfall", "dry spell", "waterlogging",
    "imd forecast", "weather forecast", "precipitation", "el nino", "la nina",
    # Agricultural climate impact
    "crop damage", "crop loss", "crop failure", "kharif", "rabi",
    "sowing delay", "harvest loss", "soil moisture", "water scarcity",
    "groundwater", "irrigation deficit", "unseasonal", "reservoir level",
    "rainfall deficiency", "normal rainfall", "southwest monsoon",
    "northeast monsoon", "kharif sowing", "water stress",
    # Pests and disease (climate-linked)
    "pest", "infestation", "locust", "blight", "rust disease",
    # Climate policy
    "climate change", "global warming", "carbon", "cop26", "cop27",
    "cop28", "ipcc", "heat stress", "skymet", "imd",
    # Lockdown as disruption (relevant for 2020)
    "lockdown",
]

COMMODITY_KEYWORDS = [
    # Vegetables
    "tomato", "onion", "potato", "garlic", "ginger", "chilli",
    "green chilli", "cabbage", "cauliflower", "brinjal", "peas",
    "carrot", "radish", "spinach", "bitter gourd", "lady finger",
    # Fruits
    "mango", "banana", "apple", "orange", "grapes", "papaya",
    "watermelon", "pomegranate",
    # Cereals & pulses
    "wheat", "rice", "paddy", "maize", "dal", "pulses", "tur dal",
    "chana", "lentil", "soybean", "jowar", "bajra",
    # Edible oils
    "mustard oil", "sunflower oil", "palm oil", "edible oil", "oilseeds",
    # Price indicators
    "mandi price", "wholesale price", "retail price", "food inflation",
    "wpi", "cpi", "price rise", "price hike", "price crash",
    "vegetable prices", "fruit prices",
    # Markets & institutions
    "agmarknet", "apmc", "mandi", "commodity market", "nafed", "fci",
    # PIB-specific agricultural terms
    "kharif crops", "rabi crops", "foodgrains", "food grains", "msp",
    "minimum support price", "procurement", "horticulture",
    "perishables", "cold storage", "warehousing", "agri",
    "farm produce", "crop production", "agricultural produce",
    "agricultural market", "farmer", "agriculture",
]

# URL-level pre-filter for Wayback/newspaper scrapers
URL_HINTS = [
    "agri", "agriculture", "commodity", "food", "price", "inflation",
    "monsoon", "flood", "drought", "crop", "market", "vegetable",
    "fruit", "weather", "climate", "mandi", "kharif", "rabi", "onion",
    "tomato", "wheat", "rice", "pulse",
]


def keyword_prefilter(text: str) -> tuple:
    """
    Strict filter for newspaper scrapers (mixed sources).
    Passes if >= 1 climate AND >= 1 commodity keyword match.
    Returns (is_relevant, climate_hits, commodity_hits).
    """
    if not text:
        return False, [], []
    t = text.lower()
    climate_hits = [k for k in CLIMATE_KEYWORDS if k in t]
    commodity_hits = [k for k in COMMODITY_KEYWORDS if k in t]
    is_relevant = len(climate_hits) >= 1 and len(commodity_hits) >= 1
    return is_relevant, climate_hits, commodity_hits


def pib_filter(text: str) -> tuple:
    """
    Relaxed filter for PIB press releases.
    PIB articles are pre-filtered by ministry so ministry selection
    already guarantees relevance. Passes if >= 1 match from either list.
    Returns (is_relevant, climate_hits, commodity_hits).
    """
    if not text or len(text.split()) < 80:
        return False, [], []
    t = text.lower()
    climate_hits = [k for k in CLIMATE_KEYWORDS if k in t]
    commodity_hits = [k for k in COMMODITY_KEYWORDS if k in t]
    is_relevant = len(climate_hits) >= 1 or len(commodity_hits) >= 1
    return is_relevant, climate_hits, commodity_hits
