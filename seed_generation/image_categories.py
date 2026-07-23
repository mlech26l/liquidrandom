"""Image category configurations: taxonomy seeds, tag vocabularies, edit palettes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditType:
    """One kind of image edit the pipeline can apply.

    The sampler draws edit types per chain (weighted, without replacement)
    and the LLM writes a concrete edit instruction for each drawn type —
    this keeps edit diversity under Python's control rather than the LLM's.
    """

    name: str
    guidance: str  # what the LLM should write a concrete instruction for
    affected_tags: tuple[str, ...] = ()  # tag attributes the edit may change
    weight: float = 1.0


@dataclass(frozen=True)
class ImageCategoryConfig:
    name: str
    display_name: str
    taxonomy_seed_prompt: str
    prompt_guidance: str  # category-specific guidance for base image prompts
    tag_attributes: dict[str, tuple[str, ...]]  # attribute -> allowed tag values
    edit_palette: tuple[EditType, ...]
    aspect_ratios: dict[str, float]  # ratio -> sampling weight
    specificity_guidance: str  # used in the VLM validation prompt


# Universal edit types for photographic categories. Category-specific edits
# below get weight 1.5 so domain-relevant variation is slightly favored.
_TIME = EditType(
    "time_of_day",
    "change the time of day (e.g. golden hour, midday, night, dawn)",
    ("time",),
)
_WEATHER = EditType(
    "weather",
    "change the weather (rain, snow, fog, overcast, bright sun)",
    ("weather",),
)
_SEASON = EditType(
    "season", "change the season, adjusting vegetation and decorations accordingly"
)
_LIGHTING = EditType(
    "lighting",
    "change the lighting (natural vs artificial, dim vs bright, colored)",
    ("lighting",),
)
_CAMERA = EditType(
    "camera_angle",
    "change the camera angle or distance (low angle, top-down, close-up, wide shot)",
)
_OBJECT_ADD = EditType("object_add", "add a plausible new object to the scene")
_OBJECT_REMOVE = EditType(
    "object_remove", "remove a prominent object from the scene"
)
_OBJECT_REPLACE = EditType(
    "object_replace", "replace one object in the scene with a different one"
)
_PEOPLE = EditType(
    "people_occupancy",
    "add or remove people, or change how many people are present",
    ("people",),
)
_DEGRADATION = EditType(
    "style_degradation",
    "apply realistic capture artifacts (motion blur, low-light noise, lens "
    "flare, slight overexposure)",
)
_COLOR = EditType(
    "color_scheme",
    "change the dominant color of key elements (repaint surfaces, recolor "
    "vehicles or products)",
)

_OUTDOOR_COMMON = (
    _TIME,
    _WEATHER,
    _SEASON,
    _LIGHTING,
    _CAMERA,
    _OBJECT_ADD,
    _OBJECT_REMOVE,
    _OBJECT_REPLACE,
    _PEOPLE,
    _DEGRADATION,
    _COLOR,
)
_INDOOR_COMMON = (
    _TIME,
    _LIGHTING,
    _CAMERA,
    _OBJECT_ADD,
    _OBJECT_REMOVE,
    _OBJECT_REPLACE,
    _PEOPLE,
    _DEGRADATION,
    _COLOR,
)

_PHOTO_RATIOS = {"4:3": 0.3, "3:2": 0.25, "16:9": 0.2, "1:1": 0.1, "9:16": 0.15}

_PHOTO_PROMPT_GUIDANCE = (
    "Prompts must describe a photorealistic photograph: name the subject, "
    "setting, lighting, time of day, camera perspective, and 2-3 distinctive "
    "concrete details. Avoid illustration or CGI styles unless the topic "
    "demands it."
)

_PEOPLE_TAGS = ("people", "no_people")


def _spec(edit: EditType) -> EditType:
    """Mark an edit type as category-specific (favored weight)."""
    return EditType(edit.name, edit.guidance, edit.affected_tags, 1.5)


IMAGE_CATEGORY_CONFIGS: dict[str, ImageCategoryConfig] = {
    "indoor_scene": ImageCategoryConfig(
        name="indoor_scene",
        display_name="Indoor Scenes",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of indoor scenes organized by building type "
            "(residential, commercial, institutional, hospitality, ...), room or "
            "space function, and style/condition. Each leaf should be a highly "
            "specific indoor scene type (e.g. 'cluttered 1970s home workshop "
            "with pegboard walls', not just 'workshop')."
        ),
        prompt_guidance=_PHOTO_PROMPT_GUIDANCE,
        tag_attributes={
            "people": _PEOPLE_TAGS,
            "setting": (
                "setting:residential",
                "setting:commercial",
                "setting:institutional",
            ),
            "lighting": ("lighting:natural", "lighting:artificial", "lighting:dim"),
            "tidiness": ("tidiness:tidy", "tidiness:cluttered"),
        },
        edit_palette=_INDOOR_COMMON
        + (
            _spec(
                EditType(
                    "tidiness",
                    "make the space significantly tidier or messier",
                    ("tidiness",),
                )
            ),
            _spec(
                EditType(
                    "renovation_style",
                    "change the interior design style (e.g. minimalist, "
                    "industrial, mid-century, rustic)",
                )
            ),
            _spec(
                EditType(
                    "furniture_change",
                    "rearrange, add, or swap major furniture pieces",
                )
            ),
        ),
        aspect_ratios=_PHOTO_RATIOS,
        specificity_guidance=(
            "The image must clearly show an indoor space matching the described "
            "room type and condition."
        ),
    ),
    "outdoor_scene": ImageCategoryConfig(
        name="outdoor_scene",
        display_name="Outdoor Scenes",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of outdoor scenes organized by environment "
            "(urban, suburban, rural, wilderness, coastal, ...), specific place "
            "type, and activity/condition. Each leaf should be a highly specific "
            "outdoor scene (e.g. 'weekend flea market in a European cobblestone "
            "square', not just 'market')."
        ),
        prompt_guidance=_PHOTO_PROMPT_GUIDANCE,
        tag_attributes={
            "people": _PEOPLE_TAGS,
            "setting": ("setting:urban", "setting:rural", "setting:nature"),
            "time": ("time:day", "time:night", "time:dawn_dusk"),
            "weather": (
                "weather:clear",
                "weather:overcast",
                "weather:rain",
                "weather:snow",
                "weather:fog",
            ),
        },
        edit_palette=_OUTDOOR_COMMON
        + (
            _spec(
                EditType(
                    "crowd_level",
                    "change how crowded the scene is (empty, sparse, busy, packed)",
                    ("people",),
                )
            ),
            _spec(
                EditType(
                    "signage",
                    "add, remove, or change signs, banners, or street markings",
                )
            ),
        ),
        aspect_ratios=_PHOTO_RATIOS,
        specificity_guidance=(
            "The image must clearly show an outdoor environment matching the "
            "described place type, weather, and time of day."
        ),
    ),
    "aerial_view": ImageCategoryConfig(
        name="aerial_view",
        display_name="Satellite & Aerial Views",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of satellite and aerial imagery organized by "
            "capture platform (satellite, drone, aircraft), terrain/land-use "
            "type, and notable features. Each leaf should be a highly specific "
            "aerial subject (e.g. 'center-pivot irrigation circles in an arid "
            "plain seen from satellite', not just 'farmland')."
        ),
        prompt_guidance=(
            "Prompts must describe a top-down or oblique aerial/satellite "
            "image: specify the platform (satellite tile, drone shot, aircraft "
            "window), altitude impression, terrain, and distinctive man-made or "
            "natural features. Realistic remote-sensing look, no maps or "
            "cartoon styles."
        ),
        tag_attributes={
            "view": ("view:satellite", "view:drone", "view:aircraft"),
            "terrain": (
                "terrain:urban",
                "terrain:agricultural",
                "terrain:forest",
                "terrain:water",
                "terrain:desert",
                "terrain:mountain",
            ),
            "time": ("time:day", "time:night"),
        },
        edit_palette=(
            _TIME,
            _SEASON,
            _CAMERA,
            _OBJECT_ADD,
            _OBJECT_REMOVE,
            _DEGRADATION,
            _spec(
                EditType(
                    "cloud_cover",
                    "add or remove cloud cover or haze over parts of the scene",
                )
            ),
            _spec(
                EditType(
                    "zoom_level",
                    "zoom in or out, changing the ground area covered",
                    ("view",),
                )
            ),
            _spec(
                EditType(
                    "night_lights",
                    "switch to a night view with artificial lights visible",
                    ("time",),
                )
            ),
            _spec(
                EditType(
                    "construction_change",
                    "add or remove buildings, roads, or construction sites",
                )
            ),
        ),
        aspect_ratios={"1:1": 0.6, "16:9": 0.2, "4:3": 0.2},
        specificity_guidance=(
            "The image must look like a genuine aerial or satellite capture of "
            "the described terrain, viewed from above."
        ),
    ),
    "agriculture": ImageCategoryConfig(
        name="agriculture",
        display_name="Agricultural Images",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of agricultural imagery organized by farming "
            "domain (crop farming, livestock, horticulture, aquaculture, ...), "
            "specific crop/animal/equipment, and context (field work, pests and "
            "disease, storage, processing). Each leaf should be highly specific "
            "(e.g. 'aphid infestation on greenhouse tomato leaves', not just "
            "'crop pests')."
        ),
        prompt_guidance=_PHOTO_PROMPT_GUIDANCE,
        tag_attributes={
            "people": _PEOPLE_TAGS,
            "subject": (
                "subject:crops",
                "subject:livestock",
                "subject:machinery",
                "subject:pests",
                "subject:infrastructure",
            ),
            "season": (
                "season:spring",
                "season:summer",
                "season:autumn",
                "season:winter",
            ),
        },
        edit_palette=_OUTDOOR_COMMON
        + (
            _spec(
                EditType(
                    "growth_stage",
                    "change the crop growth stage (seedling, mature, harvest-ready, "
                    "harvested)",
                    ("season",),
                )
            ),
            _spec(
                EditType(
                    "pest_damage",
                    "add or remove visible pest damage or plant disease symptoms",
                    ("subject",),
                )
            ),
            _spec(
                EditType(
                    "machinery",
                    "add, remove, or swap agricultural machinery in the scene",
                    ("subject",),
                )
            ),
            _spec(
                EditType(
                    "irrigation",
                    "add or change irrigation equipment or watering state",
                )
            ),
        ),
        aspect_ratios=_PHOTO_RATIOS,
        specificity_guidance=(
            "The image must clearly show the described agricultural subject "
            "(crop, animal, machine, or condition)."
        ),
    ),
    "industrial": ImageCategoryConfig(
        name="industrial",
        display_name="Factory & Industrial Images",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of factory and industrial imagery organized by "
            "industry (automotive assembly, electronics, food processing, "
            "logistics, heavy industry, ...), area (production line, warehouse, "
            "control room, loading dock), and equipment. Each leaf should be "
            "highly specific (e.g. 'robotic welding cell on an automotive "
            "chassis line', not just 'factory robot')."
        ),
        prompt_guidance=_PHOTO_PROMPT_GUIDANCE,
        tag_attributes={
            "people": _PEOPLE_TAGS,
            "subject": (
                "subject:machinery",
                "subject:robots",
                "subject:assembly_line",
                "subject:warehouse",
                "subject:exterior",
            ),
            "lighting": ("lighting:natural", "lighting:artificial", "lighting:dim"),
        },
        edit_palette=_INDOOR_COMMON
        + (
            _spec(
                EditType(
                    "worker_presence",
                    "add or remove workers, or change what they are doing",
                    ("people",),
                )
            ),
            _spec(
                EditType(
                    "machine_state",
                    "change the machine state (running, idle, under maintenance, "
                    "partially disassembled)",
                )
            ),
            _spec(
                EditType(
                    "safety_equipment",
                    "add or change safety equipment (guards, signage, PPE, barriers)",
                )
            ),
            _spec(
                EditType(
                    "clutter",
                    "change how cluttered the floor is with pallets, boxes, or tools",
                )
            ),
        ),
        aspect_ratios=_PHOTO_RATIOS,
        specificity_guidance=(
            "The image must clearly show the described industrial environment "
            "and equipment."
        ),
    ),
    "automotive": ImageCategoryConfig(
        name="automotive",
        display_name="Automotive Images",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of automotive imagery organized by viewpoint "
            "(vehicle exterior, in-cabin with driver/passengers, dashcam or road "
            "view, traffic infrastructure), vehicle type, and situation. Include "
            "a strong share of in-cabin scenes showing people inside vehicles. "
            "Each leaf should be highly specific (e.g. 'driver checking mirror "
            "in nighttime highway traffic, dashboard illuminated', not just "
            "'person in car')."
        ),
        prompt_guidance=_PHOTO_PROMPT_GUIDANCE,
        tag_attributes={
            "people": _PEOPLE_TAGS,
            "view": (
                "view:exterior",
                "view:in_cabin",
                "view:road",
                "view:traffic",
            ),
            "time": ("time:day", "time:night", "time:dawn_dusk"),
        },
        edit_palette=_OUTDOOR_COMMON
        + (
            _spec(
                EditType(
                    "cabin_occupancy",
                    "change who is inside the vehicle (driver only, passengers, "
                    "children in back, empty)",
                    ("people",),
                )
            ),
            _spec(
                EditType(
                    "traffic_density",
                    "change the traffic density on the road (empty, light, "
                    "congested)",
                )
            ),
        ),
        aspect_ratios=_PHOTO_RATIOS,
        specificity_guidance=(
            "The image must match the described automotive viewpoint (exterior, "
            "in-cabin, or road scene) and situation."
        ),
    ),
    "ui_screenshot": ImageCategoryConfig(
        name="ui_screenshot",
        display_name="UI & UX Screenshots",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of software user interfaces organized by "
            "platform (mobile app, desktop app, website, tablet, embedded/kiosk), "
            "application domain (banking, social, e-commerce, productivity, "
            "developer tools, ...), and screen type (dashboard, settings, "
            "checkout, feed, form). Each leaf should be a highly specific screen "
            "(e.g. 'flight seat-selection screen of a budget airline mobile "
            "app', not just 'travel app')."
        ),
        prompt_guidance=(
            "Prompts must describe a crisp, realistic UI screenshot: platform, "
            "app domain, screen purpose, layout regions, and realistic text "
            "labels and data values. Rendered flat UI, not a photo of a screen."
        ),
        tag_attributes={
            "platform": ("platform:mobile", "platform:desktop", "platform:tablet"),
            "theme": ("theme:light", "theme:dark"),
            "ui_state": (
                "ui_state:normal",
                "ui_state:error",
                "ui_state:empty",
                "ui_state:loading",
            ),
        },
        edit_palette=(
            _COLOR,
            _spec(
                EditType(
                    "theme_toggle",
                    "switch between light and dark theme",
                    ("theme",),
                )
            ),
            _spec(
                EditType(
                    "ui_state",
                    "change the UI state (show an error, empty state, loading "
                    "state, or success confirmation)",
                    ("ui_state",),
                )
            ),
            _spec(
                EditType(
                    "locale_translation",
                    "translate all visible text to another language, adjusting "
                    "layout naturally",
                )
            ),
            _spec(
                EditType(
                    "layout_density",
                    "change the layout density (compact vs spacious) or rearrange "
                    "main sections",
                )
            ),
            _spec(
                EditType(
                    "content_change",
                    "change the displayed data/content while keeping the same "
                    "screen design",
                )
            ),
            _spec(
                EditType(
                    "notification_overlay",
                    "add or remove a dialog, notification, or cookie banner "
                    "overlaying the screen",
                )
            ),
        ),
        aspect_ratios={"9:16": 0.45, "16:9": 0.35, "3:4": 0.1, "4:3": 0.1},
        specificity_guidance=(
            "The image must look like a real software screenshot for the "
            "described platform and screen type, with legible realistic text."
        ),
    ),
    "document": ImageCategoryConfig(
        name="document",
        display_name="Documents & OCR",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of document images organized by document type "
            "(receipts, invoices, forms, letters, handwritten notes, signage, "
            "labels, tickets, menus, certificates), domain, and medium/condition "
            "(printed, handwritten, scanned, photographed). Each leaf should be "
            "highly specific (e.g. 'crumpled thermal-paper grocery receipt with "
            "faded ink', not just 'receipt')."
        ),
        prompt_guidance=(
            "Prompts must describe a document image suitable for OCR training: "
            "document type, realistic text content and layout, medium (printed/"
            "handwritten), and capture style (flat scan or photographed at an "
            "angle). Text must be plausible and legible."
        ),
        tag_attributes={
            "medium": ("medium:printed", "medium:handwritten", "medium:mixed"),
            "capture": ("capture:scan", "capture:photo"),
            "quality": ("quality:clean", "quality:degraded"),
        },
        edit_palette=(
            _CAMERA,
            _spec(
                EditType(
                    "scan_degradation",
                    "degrade the capture quality (crumple, coffee stain, fading, "
                    "skew, shadow, low-resolution scan)",
                    ("quality",),
                )
            ),
            _spec(
                EditType(
                    "handwritten_annotation",
                    "add handwritten annotations, signatures, or margin notes",
                    ("medium",),
                )
            ),
            _spec(
                EditType(
                    "stamp_signature",
                    "add or remove official stamps, seals, or signatures",
                )
            ),
            _spec(
                EditType(
                    "locale",
                    "change the document language, adapting names, currency, and "
                    "formats",
                )
            ),
            _spec(
                EditType(
                    "content_change",
                    "change the document's data (amounts, dates, line items) while "
                    "keeping the same layout",
                )
            ),
            _spec(
                EditType(
                    "background_surface",
                    "change the surface or background the document lies on",
                    ("capture",),
                )
            ),
        ),
        aspect_ratios={"3:4": 0.6, "4:3": 0.2, "1:1": 0.1, "9:16": 0.1},
        specificity_guidance=(
            "The image must show the described document type with plausible, "
            "mostly legible text."
        ),
    ),
    "chart": ImageCategoryConfig(
        name="chart",
        display_name="Charts & Diagrams",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of charts and diagrams organized by visual form "
            "(bar/line/pie/scatter charts, flowcharts, org charts, network "
            "diagrams, tables, infographics, technical schematics), data domain "
            "(finance, science, sports, operations, ...), and complexity. Each "
            "leaf should be highly specific (e.g. 'grouped bar chart comparing "
            "quarterly revenue across three product lines', not just 'bar "
            "chart')."
        ),
        prompt_guidance=(
            "Prompts must describe a rendered chart or diagram with a concrete "
            "topic, realistic labeled axes/nodes, plausible data values, a "
            "title, and a legend where appropriate. Clean vector-like rendering "
            "unless the leaf calls for hand-drawn style."
        ),
        tag_attributes={
            "form": (
                "form:bar",
                "form:line",
                "form:pie",
                "form:scatter",
                "form:flowchart",
                "form:table",
                "form:infographic",
                "form:diagram",
            ),
            "style": ("style:clean", "style:hand_drawn"),
        },
        edit_palette=(
            _COLOR,
            _spec(
                EditType(
                    "chart_type_change",
                    "re-render the same data as a different chart type",
                    ("form",),
                )
            ),
            _spec(
                EditType(
                    "data_change",
                    "change the underlying data values while keeping the chart "
                    "type and styling",
                )
            ),
            _spec(
                EditType(
                    "annotation_add",
                    "add annotations (callouts, trend lines, highlighted regions)",
                )
            ),
            _spec(
                EditType(
                    "drawing_style",
                    "change the rendering style (clean vector, whiteboard "
                    "hand-drawn, dark dashboard theme)",
                    ("style",),
                )
            ),
            _spec(
                EditType(
                    "series_change",
                    "add or remove a data series or diagram branch, updating the "
                    "legend",
                )
            ),
        ),
        aspect_ratios={"16:9": 0.4, "4:3": 0.3, "1:1": 0.2, "3:4": 0.1},
        specificity_guidance=(
            "The image must show the described chart or diagram type with "
            "legible labels and plausible data."
        ),
    ),
    "retail_product": ImageCategoryConfig(
        name="retail_product",
        display_name="Retail & Products",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of retail and product imagery organized by "
            "product category (electronics, apparel, grocery, cosmetics, "
            "furniture, ...), presentation (studio product shot, store shelf, "
            "packaging close-up, lifestyle scene), and store format. Each leaf "
            "should be highly specific (e.g. 'supermarket dairy aisle shelf "
            "fully stocked with yogurt multipacks', not just 'supermarket')."
        ),
        prompt_guidance=_PHOTO_PROMPT_GUIDANCE,
        tag_attributes={
            "people": _PEOPLE_TAGS,
            "presentation": (
                "presentation:studio",
                "presentation:shelf",
                "presentation:packaging",
                "presentation:lifestyle",
            ),
        },
        edit_palette=_INDOOR_COMMON
        + (
            _spec(
                EditType(
                    "stock_level",
                    "change the stock level (fully stocked, half empty, sold out)",
                )
            ),
            _spec(
                EditType(
                    "promo_signage",
                    "add or remove promotional signage, price tags, or sale labels",
                )
            ),
            _spec(
                EditType(
                    "packaging_variant",
                    "change the product packaging design or variant",
                )
            ),
        ),
        aspect_ratios=_PHOTO_RATIOS,
        specificity_guidance=(
            "The image must clearly show the described product or retail "
            "presentation."
        ),
    ),
    "food": ImageCategoryConfig(
        name="food",
        display_name="Food & Cooking",
        taxonomy_seed_prompt=(
            "Generate a taxonomy of food imagery organized by cuisine, dish or "
            "ingredient type, and context (raw ingredients, cooking in progress, "
            "plated dish, kitchen scene, street food). Each leaf should be "
            "highly specific (e.g. 'hand-pulled lamian noodles being stretched "
            "in a steamy kitchen', not just 'noodles')."
        ),
        prompt_guidance=_PHOTO_PROMPT_GUIDANCE,
        tag_attributes={
            "people": _PEOPLE_TAGS,
            "stage": ("stage:raw", "stage:cooking", "stage:plated"),
            "setting": ("setting:home", "setting:restaurant", "setting:studio"),
        },
        edit_palette=_INDOOR_COMMON
        + (
            _spec(
                EditType(
                    "plating",
                    "change the plating or presentation style of the dish",
                )
            ),
            _spec(
                EditType(
                    "cooking_stage",
                    "change the cooking stage (raw, in progress, finished, eaten)",
                    ("stage",),
                )
            ),
            _spec(
                EditType(
                    "ingredient_swap",
                    "swap a main ingredient or garnish for another",
                )
            ),
        ),
        aspect_ratios=_PHOTO_RATIOS,
        specificity_guidance=(
            "The image must clearly show the described dish, ingredient, or "
            "cooking scene."
        ),
    ),
}
