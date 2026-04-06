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
    "groundwater", "irrigation deficit",
    # Climate policy
    "climate change", "global warming", "carbon", "cop26", "cop27",
    "cop28", "ipcc", "heat stress",
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
    "mustard oil", "sunflower oil", "palm oil", "edible oil",
    # Price indicators
    "mandi price", "wholesale price", "retail price", "food inflation",
    "wpi", "cpi", "price rise", "price hike", "price crash",
    "vegetable prices", "fruit prices",
    # Markets
    "agmarknet", "apmc", "mandi", "commodity market",
]

# Combined for URL-level pre-filter (short, URL-friendly words only)
URL_HINTS = [
    "agri", "agriculture", "commodity", "food", "price", "inflation",
    "monsoon", "flood", "drought", "crop", "market", "vegetable",
    "fruit", "weather", "climate", "mandi", "kharif", "rabi", "onion",
    "tomato", "wheat", "rice", "pulse",
]

def keyword_prefilter(text: str) -> tuple:
    """
    Returns (is_relevant, climate_hits, commodity_hits).
    Article passes if it has >= 1 climate AND >= 1 commodity keyword.
    """
    if not text:
        return False, [], []
    t = text.lower()
    climate_hits   = [k for k in CLIMATE_KEYWORDS   if k in t]
    commodity_hits = [k for k in COMMODITY_KEYWORDS if k in t]
    is_relevant    = len(climate_hits) >= 1 and len(commodity_hits) >= 1
    return is_relevant, climate_hits, commodity_hits
